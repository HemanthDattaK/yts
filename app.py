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

# ── API Keys ──────────────────────────────────────────────────────────────────
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "sk_ul3vhkdy_fbp9YtK")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSy7zNtAz9tn9GDbJ8")

# ── Optional: set WEBSHARE_API_KEY on Render for better proxies ───────────────
# Sign up free at https://proxy.webshare.io — 10 free proxies included
WEBSHARE_API_KEY = os.environ.get("WEBSHARE_API_KEY", "")

SARVAM_URL    = "https://api.sarvam.ai/speech-to-text"
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "downloads")
CHUNK_SECONDS = 25
MAX_RETRIES   = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL  = "gemini-2.5-flash"

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

sessions = {}

# ── In-memory proxy cache ─────────────────────────────────────────────────────
_proxy_cache: list[str] = []
_proxy_cache_time: float = 0
_PROXY_TTL = 300  # seconds


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def fmt_views(n) -> str:
    try:
        n = int(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M views"
        if n >= 1_000:     return f"{n/1_000:.1f}K views"
        return f"{n} views"
    except Exception:
        return ""


def get_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else "video"


def cleanup_video_files(video_id: str) -> None:
    seen = set()
    for pattern in [
        os.path.join(OUTPUT_DIR, f"{video_id}*.*"),
        os.path.join(OUTPUT_DIR, f"{video_id}*"),
    ]:
        for path in glob.glob(pattern):
            if path not in seen and os.path.isfile(path):
                seen.add(path)
                try:    os.remove(path)
                except: pass


# ══════════════════════════════════════════════════════════════════════════════
# Proxy helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_webshare_proxies() -> list[str]:
    """Fetch proxies from Webshare API (free tier = 10 proxies)."""
    if not WEBSHARE_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://proxy.webshare.io/api/v2/proxy/list/?mode=direct&page=1&page_size=10",
            headers={"Authorization": f"Token {WEBSHARE_API_KEY}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data    = resp.json()
        proxies = []
        for p in data.get("results", []):
            host = p.get("proxy_address", "")
            port = p.get("port", "")
            user = p.get("username", "")
            pw   = p.get("password", "")
            if host and port:
                if user and pw:
                    proxies.append(f"http://{user}:{pw}@{host}:{port}")
                else:
                    proxies.append(f"http://{host}:{port}")
        return proxies
    except Exception:
        return []


def _fetch_free_proxies() -> list[str]:
    """
    Fetch a small list of free HTTP proxies from proxyscrape.
    These are unreliable but free — used as last resort when no Webshare key.
    """
    try:
        resp = requests.get(
            "https://api.proxyscrape.com/v3/free-proxy-list/get"
            "?request=displayproxies&protocol=http&timeout=5000"
            "&country=US,GB,DE&ssl=yes&anonymity=elite",
            timeout=10,
        )
        lines   = [l.strip() for l in resp.text.splitlines() if l.strip()]
        proxies = [f"http://{l}" for l in lines[:8]]
        return proxies
    except Exception:
        return []


def get_proxies() -> list[str]:
    """Return cached proxy list, refreshing every _PROXY_TTL seconds."""
    global _proxy_cache, _proxy_cache_time
    now = time.time()
    if _proxy_cache and now - _proxy_cache_time < _PROXY_TTL:
        return _proxy_cache

    proxies = _fetch_webshare_proxies()
    if not proxies:
        proxies = _fetch_free_proxies()

    _proxy_cache      = proxies
    _proxy_cache_time = now
    return proxies


# ══════════════════════════════════════════════════════════════════════════════
# YouTube search
# ══════════════════════════════════════════════════════════════════════════════

def search_youtube(query: str, top_n: int = 2) -> list[dict]:
    ydl_opts = {
        "quiet":        True,
        "no_warnings":  True,
        "extract_flat": "in_playlist",
        "force_ipv4":   True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{top_n}:{query}", download=False)

    results = []
    for entry in info.get("entries", []):
        vid_id     = entry.get("id", "")
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


# ══════════════════════════════════════════════════════════════════════════════
# Audio download — tries proxies to bypass Render IP ban
# ══════════════════════════════════════════════════════════════════════════════

def _base_ydl_opts(vid: str, proxy: str | None = None) -> dict:
    outtmpl = os.path.join(OUTPUT_DIR, f"{vid}.%(ext)s")
    opts = {
        "format":           "bestaudio/best",
        "outtmpl":          outtmpl,
        "quiet":            True,
        "no_warnings":      True,
        "socket_timeout":   30,
        "retries":          5,
        "fragment_retries": 5,
        "extractor_args":   {"youtube": {"player_client": ["tv_embedded", "web_creator"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }
    if proxy:
        opts["proxy"] = proxy
    return opts


def _find_downloaded(vid: str) -> str | None:
    candidates = []
    for ext in ("mp4", "m4a", "webm", "mkv", "opus", "mp3", "ogg"):
        candidates.extend(glob.glob(os.path.join(OUTPUT_DIR, f"{vid}*.{ext}")))
    candidates = [p for p in candidates if not p.endswith(".wav")]
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    return None


def download_audio(url: str) -> str | None:
    vid = get_video_id(url)

    # ── Attempt 1: no proxy (sometimes works on fresh deploys) ───────────────
    try:
        with yt_dlp.YoutubeDL(_base_ydl_opts(vid)) as ydl:
            ydl.download([url])
        result = _find_downloaded(vid)
        if result:
            return result
    except Exception:
        pass

    # ── Attempts 2+: rotate through proxies ──────────────────────────────────
    proxies = get_proxies()
    for proxy in proxies:
        try:
            with yt_dlp.YoutubeDL(_base_ydl_opts(vid, proxy)) as ydl:
                ydl.download([url])
            result = _find_downloaded(vid)
            if result:
                return result
        except Exception:
            time.sleep(1)
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# WAV conversion + chunking
# ══════════════════════════════════════════════════════════════════════════════

def convert_to_wav(input_file: str) -> str | None:
    base, _  = os.path.splitext(input_file)
    wav_file = base + "_16k.wav"
    result   = subprocess.run(
        ["ffmpeg", "-y", "-i", input_file,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_file],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not os.path.isfile(wav_file):
        return None
    return wav_file


def get_duration(wav_file: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", wav_file],
        capture_output=True, text=True
    )
    try:    return float(result.stdout.strip())
    except: return 0.0


def split_wav(wav_file: str) -> list[str]:
    duration = get_duration(wav_file)
    if not duration:
        return [wav_file]
    n_chunks = math.ceil(duration / CHUNK_SECONDS)
    if n_chunks == 1:
        return [wav_file]
    base, _     = os.path.splitext(wav_file)
    chunk_paths = []
    for i in range(n_chunks):
        chunk_path = f"{base}_chunk{i:03d}.wav"
        r = subprocess.run(
            ["ffmpeg", "-y",
             "-ss", str(i * CHUNK_SECONDS), "-t", str(CHUNK_SECONDS),
             "-i",  wav_file,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", chunk_path],
            capture_output=True, text=True
        )
        if r.returncode == 0 and os.path.isfile(chunk_path):
            chunk_paths.append(chunk_path)
    return chunk_paths or [wav_file]


# ══════════════════════════════════════════════════════════════════════════════
# Sarvam transcription
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_chunk(audio_file: str) -> str:
    with open(audio_file, "rb") as f:
        resp = requests.post(
            SARVAM_URL,
            headers={"api-subscription-key": SARVAM_API_KEY},
            data={"model": "saarika:v2.5"},
            files={"file": (os.path.basename(audio_file), f, "audio/wav")},
            timeout=120,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Sarvam {resp.status_code}: {resp.text}")
    return resp.json().get("transcript", "")


def transcribe_audio(wav_file: str) -> str:
    chunks = split_wav(wav_file)
    parts  = []
    for chunk in chunks:
        try:
            parts.append(transcribe_chunk(chunk))
        except RuntimeError:
            parts.append("")
        finally:
            if chunk != wav_file and os.path.isfile(chunk):
                try:    os.remove(chunk)
                except: pass
    return " ".join(parts)


def get_transcript_for_video(video: dict) -> str | None:
    video_id = video["video_id"]
    audio = wav = None
    try:
        audio = download_audio(video["url"])
        if not audio:
            return None
        wav = convert_to_wav(audio)
        if not wav:
            return None
        return transcribe_audio(wav)
    finally:
        for f in [wav, audio]:
            if f and os.path.isfile(f):
                try:    os.remove(f)
                except: pass
        cleanup_video_files(video_id)


# ══════════════════════════════════════════════════════════════════════════════
# SSE helper
# ══════════════════════════════════════════════════════════════════════════════

def sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"


# ══════════════════════════════════════════════════════════════════════════════
# /api/search
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/search", methods=["POST"])
def api_search():
    body       = request.json or {}
    topic      = body.get("topic", "").strip()
    question   = body.get("question", "").strip() or f"Give a detailed summary and key insights about: {topic}"
    num_videos = max(1, min(10, int(body.get("num_videos", 2))))
    language   = body.get("language", "English")
    session_id = body.get("session_id", "default")

    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    def generate():
        yield sse("status", {"message": f"Searching YouTube for '{topic}'..."})
        try:
            videos = search_youtube(topic, top_n=num_videos)
        except Exception as e:
            yield sse("error", {"message": str(e)}); return

        if not videos:
            yield sse("error", {"message": "No videos found."}); return

        yield sse("videos", {"videos": videos})

        transcripts = []
        for idx, video in enumerate(videos):
            yield sse("status", {"message": f"Downloading & transcribing video {idx+1}/{len(videos)}: {video['title'][:50]}..."})
            text = get_transcript_for_video(video)
            if text:
                transcripts.append({**video, "text": text})
                yield sse("transcript_ready", {"video_id": video["video_id"], "title": video["title"]})
                yield sse("transcript_text",  {"video_id": video["video_id"], "text": text})
            else:
                yield sse("transcript_error", {"video_id": video["video_id"], "title": video["title"]})

        if not transcripts:
            yield sse("error", {"message": "Could not transcribe any videos."}); return

        sessions[session_id] = {
            "topic":       topic,
            "transcripts": transcripts,
            "history":     [{"role": "user", "content": question}],
        }

        yield sse("status", {"message": "Generating AI answer..."})
        yield sse("answer_start", {})

        context = "\n\n".join(
            f"--- SOURCE {i}: {t['title']} ---\nURL: {t['url']}\n\n{t['text']}"
            for i, t in enumerate(transcripts, 1)
        )
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
            for chunk in gemini_client.models.generate_content_stream(model=GEMINI_MODEL, contents=prompt):
                word = chunk.text or ""
                full_answer += word
                yield sse("token", {"text": word})
            sessions[session_id]["history"].append({"role": "assistant", "content": full_answer})
            yield sse("answer_done", {"full_answer": full_answer})
        except Exception as e:
            yield sse("error", {"message": f"Gemini error: {str(e)}"})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════════════════════════
# /api/followup
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
        context = "\n\n".join(
            f"--- SOURCE {i}: {t['title']} ---\nURL: {t['url']}\n\n{t['text']}"
            for i, t in enumerate(session["transcripts"], 1)
        )
        history_text = "".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\n\n"
            for m in session["history"]
        )
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
            for chunk in gemini_client.models.generate_content_stream(model=GEMINI_MODEL, contents=prompt):
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
# /api/add_videos
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/add_videos", methods=["POST"])
def api_add_videos():
    body       = request.json or {}
    query      = body.get("topic", "").strip()
    num_videos = max(1, min(5, int(body.get("num_videos", 1))))
    session_id = body.get("session_id", "default")

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "No active session."}), 400
    if not query:
        return jsonify({"error": "No query provided"}), 400

    existing_ids = {t["video_id"] for t in session["transcripts"]}

    def generate():
        yield sse("status", {"message": f"Searching for more videos: '{query}'..."})
        try:
            candidates = search_youtube(query, top_n=num_videos + 3)
        except Exception as e:
            yield sse("error", {"message": str(e)}); return

        new_videos = [v for v in candidates if v["video_id"] not in existing_ids][:num_videos]
        if not new_videos:
            yield sse("error", {"message": "No new videos found."}); return

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
                yield sse("transcript_text",  {"video_id": video["video_id"], "text": text})
            else:
                yield sse("transcript_error", {"video_id": video["video_id"], "title": video["title"]})

        yield sse("done", {"count": len(added), "message": f"Added {len(added)} new video(s)."})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
