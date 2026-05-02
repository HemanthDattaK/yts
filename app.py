import os
import re
import glob
import math
import time
import json
import subprocess
import requests
import yt_dlp
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from google import genai

# ── API Keys ───────────────────────────────────────────────────────────────────
SARVAM_API_KEY = "sk_zd8flmdq_ScDU3PJg8VRUSwIR9d6bUMD1"
GEMINI_API_KEY = "AIzaSyA7R0Gj-wrg4ZsrF85i1ELE4DBypGnKhas"

SARVAM_URL    = "https://api.sarvam.ai/speech-to-text"
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "downloads")
CHUNK_SECONDS = 25
MAX_RETRIES   = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL  = "gemini-2.5-flash"

app = Flask(__name__, static_folder="static")
CORS(app)

# ── In-memory chat sessions ────────────────────────────────────────────────────
# { session_id: { "topic": str, "transcripts": [...], "history": [...] } }
sessions = {}

# ══════════════════════════════════════════════════════════════════════════════
# YouTube helpers
# ══════════════════════════════════════════════════════════════════════════════
def search_youtube(query: str, top_n: int = 2) -> list[dict]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "force_ipv4": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{top_n}:{query}", download=False)

    results = []
    for entry in info.get("entries", []):
        vid_id = entry.get("id", "")
        view_count = entry.get("view_count", 0)
        results.append({
            "title":      entry.get("title", "Unknown"),
            "url":        f"https://www.youtube.com/watch?v={vid_id}",
            "video_id":   vid_id,
            "thumbnail":  f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg",
            "channel":    entry.get("uploader", entry.get("channel", "Unknown")),
            "duration":   entry.get("duration_string", entry.get("duration", "")),
            "view_count": view_count,
            "views_fmt":  fmt_views(view_count),
        })
    return results

def fmt_views(n) -> str:
    try:
        n = int(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M views"
        if n >= 1_000:     return f"{n/1_000:.1f}K views"
        return f"{n} views"
    except:
        return ""

def get_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else "video"

def download_audio(url: str) -> str | None:
    vid = get_video_id(url)
    outtmpl = os.path.join(OUTPUT_DIR, f"{vid}.%(ext)s")
    ydl_opts = {
        "format": "18/bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "force_ipv4": True,
        "geo_bypass": True,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "http_chunk_size": 1048576,
        "extractor_args": {"youtube": {"player_client": ["android", "web", "ios"]}},
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            candidates = []
            for ext in ("mp4", "m4a", "webm", "mkv", "opus", "mp3"):
                candidates.extend(glob.glob(os.path.join(OUTPUT_DIR, f"{vid}*.{ext}")))
            candidates = [p for p in candidates if not p.endswith(".wav")]
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return candidates[0]
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 5)
    return None

def convert_to_wav(input_file: str) -> str | None:
    base, _ = os.path.splitext(input_file)
    wav_file = base + "_16k.wav"
    cmd = ["ffmpeg", "-y", "-i", input_file,
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(wav_file):
        return None
    return wav_file

def get_duration(wav_file: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", wav_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

def split_wav(wav_file: str) -> list[str]:
    duration = get_duration(wav_file)
    if duration == 0:
        return [wav_file]
    n_chunks = math.ceil(duration / CHUNK_SECONDS)
    if n_chunks == 1:
        return [wav_file]
    base, _ = os.path.splitext(wav_file)
    chunk_paths = []
    for i in range(n_chunks):
        chunk_path = f"{base}_chunk{i:03d}.wav"
        cmd = ["ffmpeg", "-y", "-ss", str(i * CHUNK_SECONDS), "-t", str(CHUNK_SECONDS),
               "-i", wav_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", chunk_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.isfile(chunk_path):
            chunk_paths.append(chunk_path)
    return chunk_paths or [wav_file]

def transcribe_chunk(audio_file: str) -> str:
    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(audio_file, "rb") as f:
        resp = requests.post(
            SARVAM_URL,
            headers=headers,
            data={"model": "saarika:v2.5"},
            files={"file": (os.path.basename(audio_file), f, "audio/wav")},
            timeout=120,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Sarvam {resp.status_code}: {resp.text}")
    return resp.json().get("transcript", "")

def transcribe_audio(wav_file: str) -> str:
    chunks = split_wav(wav_file)
    parts = []
    for chunk in chunks:
        try:
            parts.append(transcribe_chunk(chunk))
        except RuntimeError:
            parts.append("")
        finally:
            if chunk != wav_file and os.path.isfile(chunk):
                os.remove(chunk)
    return " ".join(parts)

def get_transcript_for_video(video: dict) -> str | None:
    audio = download_audio(video["url"])
    if not audio:
        return None
    wav = convert_to_wav(audio)
    if not wav:
        return None
    return transcribe_audio(wav)

# ══════════════════════════════════════════════════════════════════════════════
# SSE helper
# ══════════════════════════════════════════════════════════════════════════════
def sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"

# ══════════════════════════════════════════════════════════════════════════════
# /api/search  — new topic search
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/search", methods=["POST"])
def api_search():
    body        = request.json or {}
    topic       = body.get("topic", "").strip()
    question    = body.get("question", "").strip() or f"Give a detailed summary and key insights about: {topic}"
    num_videos  = max(1, min(10, int(body.get("num_videos", 2))))
    language    = body.get("language", "English")
    session_id  = body.get("session_id", "default")

    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    def generate():
        yield sse("status", {"message": f"Searching YouTube for '{topic}'..."})
        try:
            videos = search_youtube(topic, top_n=num_videos)
        except Exception as e:
            yield sse("error", {"message": str(e)})
            return

        if not videos:
            yield sse("error", {"message": "No videos found."})
            return

        yield sse("videos", {"videos": videos})

        transcripts = []
        for idx, video in enumerate(videos):
            yield sse("status", {"message": f"Downloading & transcribing video {idx+1}/{len(videos)}: {video['title'][:50]}..."})
            text = get_transcript_for_video(video)
            if text:
                transcripts.append({**video, "text": text})
                yield sse("transcript_ready", {"video_id": video["video_id"], "title": video["title"]})
                # Send the actual transcript text so frontend can show it
                yield sse("transcript_text", {"video_id": video["video_id"], "text": text})
            else:
                yield sse("transcript_error", {"video_id": video["video_id"], "title": video["title"]})

        if not transcripts:
            yield sse("error", {"message": "Could not transcribe any videos."})
            return

        sessions[session_id] = {
            "topic": topic,
            "transcripts": transcripts,
            "history": [{"role": "user", "content": question}],
        }

        yield sse("status", {"message": "Generating AI answer..."})
        yield sse("answer_start", {})

        context_blocks = []
        for i, t in enumerate(transcripts, 1):
            context_blocks.append(
                f"--- SOURCE {i}: {t['title']} ---\nURL: {t['url']}\n\n{t['text']}"
            )
        context = "\n\n".join(context_blocks)

        prompt = f"""You are an expert research assistant. The user searched for "{topic}" on YouTube.
Below are transcripts from the top {len(transcripts)} videos.

{context}

---
Answer the following question clearly and in detail. Respond in {language}.

Question: {question}

If transcripts lack info, say so and summarize what IS available.
Use clear headings (##) for long responses. Be thorough and helpful.
"""

        try:
            full_answer = ""
            for chunk in gemini_client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=prompt,
            ):
                word = chunk.text or ""
                full_answer += word
                yield sse("token", {"text": word})

            sessions[session_id]["history"].append({"role": "assistant", "content": full_answer})
            yield sse("answer_done", {"full_answer": full_answer})

            safe_topic = re.sub(r'[\\/*?:"<>|]', "_", topic)[:40]
            answer_path = os.path.join(OUTPUT_DIR, f"{safe_topic}_answer.txt")
            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(f"Topic: {topic}\nQuestion: {question}\n\n")
                for t in transcripts:
                    f.write(f"Source: {t['title']} — {t['url']}\n")
                f.write("\n" + "="*60 + "\n\n" + full_answer)
            yield sse("saved", {"path": answer_path})

        except Exception as e:
            yield sse("error", {"message": f"Gemini error: {str(e)}"})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════════════════════════
# /api/followup  — follow-up question (NO re-search)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/followup", methods=["POST"])
def api_followup():
    body       = request.json or {}
    question   = body.get("question", "").strip()
    language   = body.get("language", "English")
    session_id = body.get("session_id", "default")

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "No active session. Please search for a topic first."}), 400
    if not question:
        return jsonify({"error": "No question provided"}), 400

    def generate():
        yield sse("answer_start", {})

        context_blocks = []
        for i, t in enumerate(session["transcripts"], 1):
            context_blocks.append(
                f"--- SOURCE {i}: {t['title']} ---\nURL: {t['url']}\n\n{t['text']}"
            )
        context = "\n\n".join(context_blocks)

        history_text = ""
        for msg in session["history"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"

        prompt = f"""You are an expert research assistant continuing a conversation about "{session['topic']}".

VIDEO TRANSCRIPTS:
{context}

CONVERSATION HISTORY:
{history_text}

Now answer this follow-up question in {language}:
{question}

Use the transcripts and conversation history. Use headings if needed.
"""

        session["history"].append({"role": "user", "content": question})

        try:
            full_answer = ""
            for chunk in gemini_client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=prompt,
            ):
                word = chunk.text or ""
                full_answer += word
                yield sse("token", {"text": word})

            session["history"].append({"role": "assistant", "content": full_answer})
            yield sse("answer_done", {"full_answer": full_answer})

        except Exception as e:
            yield sse("error", {"message": f"Gemini error: {str(e)}"})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════════════════════════
# /api/add_videos  — add more videos to an existing session
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/add_videos", methods=["POST"])
def api_add_videos():
    body       = request.json or {}
    query      = body.get("topic", "").strip()
    num_videos = max(1, min(5, int(body.get("num_videos", 1))))
    session_id = body.get("session_id", "default")

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "No active session. Please search for a topic first."}), 400
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Get video IDs already in session to avoid duplicates
    existing_ids = {t["video_id"] for t in session["transcripts"]}

    def generate():
        yield sse("status", {"message": f"Searching for more videos: '{query}'..."})

        try:
            # Fetch a few extra to account for duplicates
            candidates = search_youtube(query, top_n=num_videos + 3)
        except Exception as e:
            yield sse("error", {"message": str(e)})
            return

        # Filter out already-loaded videos
        new_videos = [v for v in candidates if v["video_id"] not in existing_ids][:num_videos]

        if not new_videos:
            yield sse("error", {"message": "No new videos found (all results already loaded)."})
            return

        yield sse("videos", {"videos": new_videos})

        added = []
        for idx, video in enumerate(new_videos):
            yield sse("status", {"message": f"Downloading & transcribing {idx+1}/{len(new_videos)}: {video['title'][:50]}..."})
            text = get_transcript_for_video(video)
            if text:
                entry = {**video, "text": text}
                session["transcripts"].append(entry)
                added.append(entry)
                yield sse("transcript_ready", {"video_id": video["video_id"], "title": video["title"]})
                yield sse("transcript_text", {"video_id": video["video_id"], "text": text})
            else:
                yield sse("transcript_error", {"video_id": video["video_id"], "title": video["title"]})

        yield sse("done", {
            "count": len(added),
            "message": f"Added {len(added)} new video(s) to the session."
        })

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)