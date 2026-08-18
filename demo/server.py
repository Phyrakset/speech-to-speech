"""
Tiny server for the speech-to-speech demo.

The demo used to ship as a `sdk: static` Space, but the web-search tool needs a
search key the browser must NOT see. A static Space has no runtime process, so it
can't hold a secret the front-end uses. This server fixes that: it serves the
unchanged front-end AND exposes a same-origin `/api/search` proxy that holds the
Serper key server-side (see docs/adr/0001).

Everything lives in one container; the speech-to-speech backend stays a separate,
load-balanced service the browser talks to over WebSocket as before. The load
balancer's address is a secret too (like the Serper key): the browser never sees
it. `/api/session` proxies the session handshake server-side so only the
per-session compute URL the LB hands back (which the browser must dial) is exposed.

On the deployed Space the server also meters conversation time by HF login tier
(anonymous / signed-in / PRO) — see `limiter.py` and `auth.py`. That whole feature
is off unless BOTH `LOAD_BALANCER_URL` and `SPACE_ID` are set, so it runs only on
the live Space, never locally (even with the LB exported for testing).

`SPEECH_TO_SPEECH_URL` overrides everything: when set, the LB logic above is
disabled entirely (no session proxy, no queue, no metering, no sign-in) and the
browser connects directly to that URL, shown read-only in Settings.

Endpoints:
  GET  /api/config           -> { search, lb, allowDirect, s2sUrl, rtc, iceServers, auth }
  GET  /api/me               -> login + tier + remaining budget (LB mode only)
  POST /api/search           -> { results, answer }  Google via Serper.dev
  POST /api/calls            -> proxies the WebRTC SDP offer to <s2s>/v1/realtime/calls
  POST /api/session          -> proxies <LB>/session: a grant, or a queue ticket
  GET  /api/queue/{id}       -> proxies <LB>/queue/{id}: position, or a grant on claim
  DELETE /api/queue/{id}     -> leave the queue (explicit "Leave queue" button)
  POST /api/queue/end        -> leave the queue (sendBeacon on teardown)
  POST /api/session/heartbeat-> extend the reservation; { expired }
  POST /api/session/end      -> reconcile + refund (sendBeacon on teardown)
  /*                         -> static files (index.html, main.js, ...)

When every compute slot is busy the load balancer hands back a queue ticket
instead of a grant; the browser polls /api/queue/{id} until it reaches the front
and a slot frees. Waiting reserves nothing — the daily budget is only reserved at
the moment a slot is actually claimed (a grant), never while queued.
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import auth
import httpx
import limiter
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from speech_to_speech.TTS.voice_cloning_bench import VoiceCloningService
from pipeline_manager import PipelineManager, PipelineConfig

logger = logging.getLogger("s2s.search")

SERPER_KEY = os.environ.get("SERPER_API_KEY", "").strip()
# Speech-to-speech load balancer URL. When set, the browser POSTs /api/session
# (which proxies <lb>/session here, server-side) and connects to the URL the LB
# returns (the original flow). The LB address itself is never sent to the browser.
# When empty, the user may instead set a direct s2s server URL in Settings and the
# browser connects to it straight (no load balancer).
LOAD_BALANCER_URL = os.environ.get("LOAD_BALANCER_URL", "").strip()
# Direct s2s server URL pinned by the deploy. Takes priority over the load
# balancer: when set, ALL LB logic is disabled (no /api/session proxy, no queue,
# no limiter, no sign-in) and the browser connects to this URL directly. Unlike
# the LB address it is NOT a secret — /api/config sends it to the client, which
# shows it read-only in Settings.
SPEECH_TO_SPEECH_URL = os.environ.get("SPEECH_TO_SPEECH_URL", "").strip()
if SPEECH_TO_SPEECH_URL:
    LOAD_BALANCER_URL = ""
# HF injects SPACE_ID ("owner/space") into every Space runtime; it's absent
# locally and on a plain `docker run`. We meter conversation time ONLY on the
# deployed Space — i.e. when BOTH the LB is configured AND we're on a Space.
# Off-Space (local dev, even with the LB exported) the app still proxies the LB,
# but nothing is metered: no budget, no reservations, no sign-in gating.
SPACE_ID = os.environ.get("SPACE_ID", "").strip()
LIMITER_ENABLED = bool(LOAD_BALANCER_URL) and bool(SPACE_ID)


def _parse_ice_servers(raw: str) -> list:
    """ICE servers for the browser's RTCPeerConnection, from RTC_ICE_SERVERS.

    Accepts a JSON list of RTCIceServer dicts (same format as the s2s
    server's SPEECH_TO_SPEECH_ICE_SERVERS, e.g.
    ``[{"urls": "turn:t.example.com", "username": "u", "credential": "c"}]``),
    a single such dict, or a plain comma-separated list of STUN/TURN URLs.
    Empty when unset — host candidates only, which is fine for local use."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except ValueError:
        pass
    return [{"urls": u.strip()} for u in raw.split(",") if u.strip()]


RTC_ICE_SERVERS = _parse_ice_servers(os.environ.get("RTC_ICE_SERVERS", ""))
DEFAULT_STARTUP_GREETING = (
    "Start the conversation now with a brief, spontaneous greeting in character. "
    "Keep it to one sentence, invite the user in naturally, and vary the wording each time."
)
# Exposed to the browser through /api/config. Set an empty value to disable the
# automatic greeting without changing the client bundle.
STARTUP_GREETING = os.environ.get("STARTUP_GREETING", DEFAULT_STARTUP_GREETING).strip()


def _webrtc_calls_url(s2s_url: str) -> str:
    """Derive the WebRTC handshake URL from the pinned realtime URL.

    ``ws://host:port/v1/realtime`` -> ``http://host:port/v1/realtime/calls``
    (ws->http, wss->https; a bare host gets the default /v1/realtime path,
    mirroring the client's buildDirectWsUrl normalisation)."""
    s = s2s_url.strip()
    if not s.startswith(("ws://", "wss://", "http://", "https://")):
        s = "http://" + s
    parts = urlsplit(s)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    path = parts.path if parts.path not in ("", "/") else "/v1/realtime"
    return urlunsplit((scheme, parts.netloc, path.rstrip("/") + "/calls", parts.query, ""))


SERPER_URL = "https://google.serper.dev/search"
# Cap results so the tool output stays small enough to feed back to the model.
MAX_RESULTS = 5
HERE = os.path.dirname(os.path.abspath(__file__))
LB_USER_AGENT = "speech-to-speech-demo"

app = FastAPI(title="s2s-demo")


@app.get("/vendor/openai-realtime-agents.umd.js", include_in_schema=False)
def agents_sdk_bundle():
    """Serve the exact npm-pinned browser bundle without committing build output."""
    bundled = os.path.join(HERE, "vendor", "openai-realtime-agents.umd.js")
    installed = os.path.join(
        HERE,
        "node_modules",
        "@openai",
        "agents-realtime",
        "dist",
        "bundle",
        "openai-realtime-agents.umd.js",
    )
    path = bundled if os.path.isfile(bundled) else installed
    if not os.path.isfile(path):
        raise HTTPException(status_code=503, detail="Run npm ci in demo/")
    return FileResponse(path, media_type="text/javascript")

# Wire HF OAuth before the app serves (no-op unless the OAuth env is present).
# Sign-in only matters when we're metering (prod Space), so gate it on that.
AUTH_ENABLED = LIMITER_ENABLED and auth.attach(app)


@app.on_event("startup")
async def _startup():
    """Stand up the usage DB and a periodic sweeper — metered (prod Space) only."""
    if LIMITER_ENABLED:
        await asyncio.to_thread(limiter.start)


@app.on_event("shutdown")
def _shutdown():
    """Tear down limiter and background pipeline subprocess."""
    if LIMITER_ENABLED:
        limiter.stop()
    PipelineManager.get_instance().stop()


class SearchRequest(BaseModel):
    query: str
    key: Optional[str] = None


@app.get("/api/config")
def config():
    """Client bootstrap config and pipeline status."""
    pipe_status = PipelineManager.get_instance().get_status()
    # Default direct s2s url points to local pipeline if none pinned
    effective_s2s_url = SPEECH_TO_SPEECH_URL or f"ws://127.0.0.1:{pipe_status['port']}/v1/realtime"
    return {
        "search": bool(SERPER_KEY),
        "lb": bool(LOAD_BALANCER_URL),
        "allowDirect": not LOAD_BALANCER_URL,
        "s2sUrl": effective_s2s_url,
        "rtc": bool(SPEECH_TO_SPEECH_URL),
        "iceServers": RTC_ICE_SERVERS,
        "startupGreeting": STARTUP_GREETING,
        "auth": AUTH_ENABLED,
        "pipeline": pipe_status,
    }


UPLOADS_DIR = Path(HERE) / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Auto-copy reference sounds from assets/reference_sound/
_ASSETS_REF_DIR = Path(HERE).parent / "assets" / "reference_sound"
if _ASSETS_REF_DIR.is_dir():
    for ext in ("*.wav", "*.mp3", "*.ogg", "*.flac"):
        for ref_f in _ASSETS_REF_DIR.glob(ext):
            dest_f = UPLOADS_DIR / ref_f.name
            if not dest_f.is_file():
                try:
                    shutil.copy(ref_f, dest_f)
                    logger.info("Copied asset reference voice: %s", ref_f.name)
                except Exception as e:
                    logger.warning("Could not copy reference file %s: %s", ref_f.name, e)


@app.get("/api/providers")
def get_providers():
    """Return available STT, LLM, and TTS providers, configured defaults from .env, and voice options."""
    cloner = VoiceCloningService.get_instance()
    info = cloner.get_providers_info()
    # List available uploaded/preset reference files
    ref_files = []
    seen = set()
    for ext in ("*.wav", "*.mp3", "*.ogg", "*.flac"):
        for p in sorted(UPLOADS_DIR.glob(ext)):
            if p.name in seen:
                continue
            seen.add(p.name)
            is_valid, dur, _ = cloner.validate_reference_audio(p)
            if is_valid:
                ref_files.append({"name": p.name, "duration": round(dur, 2)})
    info["reference_files"] = ref_files
    info["pipeline_status"] = PipelineManager.get_instance().get_status()
    info["env_defaults"] = {
        "stt_provider": os.getenv("DEFAULT_STT_PROVIDER", "parakeet-tdt"),
        "llm_provider": os.getenv("DEFAULT_LLM_PROVIDER", "gemini-flash"),
        "llm_model": os.getenv("MODEL_NAME", os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")),
        "tts_provider": os.getenv("DEFAULT_TTS_PROVIDER", "qwen3"),
        "tts_model": os.getenv("DEFAULT_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        "tts_backend": os.getenv("DEFAULT_TTS_BACKEND", "torch"),
        "ref_audio_name": os.getenv("DEFAULT_REFERENCE_VOICE", "khmer_bong_nika_sound.wav"),
        "language": os.getenv("DEFAULT_TTS_LANGUAGE", "auto"),
        "port": int(os.getenv("PIPELINE_PORT", "8081")),
    }
    return info


class PipelineStartRequest(BaseModel):
    stt_provider: str = Field(default_factory=lambda: os.getenv("DEFAULT_STT_PROVIDER", "parakeet-tdt"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "gemini-flash"))
    tts_provider: str = Field(default_factory=lambda: os.getenv("DEFAULT_TTS_PROVIDER", "qwen3"))
    tts_model_name: str = Field(default_factory=lambda: os.getenv("DEFAULT_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    tts_backend: str = Field(default_factory=lambda: os.getenv("DEFAULT_TTS_BACKEND", "torch"))
    ref_audio_name: Optional[str] = Field(default_factory=lambda: os.getenv("DEFAULT_REFERENCE_VOICE", "khmer_bong_nika_sound.wav"))
    ref_transcript: Optional[str] = None
    xvec_only: bool = True
    language: str = Field(default_factory=lambda: os.getenv("DEFAULT_TTS_LANGUAGE", "auto"))
    port: int = Field(default_factory=lambda: int(os.getenv("PIPELINE_PORT", "8081")))


@app.post("/api/pipeline/start")
def start_pipeline(req: PipelineStartRequest):
    """Launch the speech-to-speech serve subprocess with user-configured providers and voice."""
    ref_audio_path = None
    if req.ref_audio_name:
        candidate = UPLOADS_DIR / req.ref_audio_name
        if candidate.is_file():
            ref_audio_path = str(candidate)
        else:
            # Check assets directly
            asset_candidate = _ASSETS_REF_DIR / req.ref_audio_name
            if asset_candidate.is_file():
                ref_audio_path = str(asset_candidate)

    cfg = PipelineConfig(
        stt_provider=req.stt_provider,
        llm_provider=req.llm_provider,
        tts_provider=req.tts_provider,
        tts_model_name=req.tts_model_name,
        tts_backend=req.tts_backend,
        ref_audio_path=ref_audio_path,
        ref_transcript=req.ref_transcript,
        xvec_only=req.xvec_only,
        language=req.language,
        port=req.port,
    )

    manager = PipelineManager.get_instance()
    res = manager.start(cfg)
    if not res.get("ok"):
        raise HTTPException(status_code=500, detail=res.get("error", "Failed to start pipeline"))
    return res


@app.post("/api/pipeline/stop")
def stop_pipeline():
    """Stop the background speech-to-speech pipeline."""
    manager = PipelineManager.get_instance()
    return manager.stop()


@app.get("/api/pipeline/status")
def pipeline_status():
    """Get current status of the background speech-to-speech pipeline."""
    return PipelineManager.get_instance().get_status()


@app.get("/api/pipeline/logs")
def pipeline_logs():
    """Get recent logs from the speech-to-speech pipeline."""
    manager = PipelineManager.get_instance()
    status = manager.get_status()
    return {
        "running": status["running"],
        "logs": status["recent_logs"],
    }


@app.post("/api/tts/upload-reference")
async def upload_reference_audio(file: UploadFile = File(...)):
    """Upload a reference audio file for voice cloning."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    # Sanitize filename
    clean_name = Path(file.filename).name.replace(" ", "_")
    if not (clean_name.endswith(".wav") or clean_name.endswith(".mp3") or clean_name.endswith(".ogg") or clean_name.endswith(".flac")):
        raise HTTPException(status_code=400, detail="Only audio files (.wav, .mp3, .ogg, .flac) are supported.")

    dest_path = UPLOADS_DIR / clean_name
    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    cloner = VoiceCloningService.get_instance()
    is_valid, dur, err_msg = cloner.validate_reference_audio(dest_path)
    if not is_valid:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {err_msg}")

    return {
        "ok": True,
        "filename": clean_name,
        "duration": round(dur, 2),
        "url": f"/api/tts/reference-audio/{clean_name}",
    }


@app.get("/api/tts/reference-audio/{filename}")
def get_reference_audio(filename: str):
    """Serve uploaded reference audio for playback."""
    clean_name = Path(filename).name
    path = UPLOADS_DIR / clean_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference audio file not found.")
    media_type = "audio/wav" if clean_name.endswith(".wav") else "audio/mpeg"
    return FileResponse(str(path), media_type=media_type)


@app.post("/api/tts/clone")
async def clone_voice(
    text: str = Form(...),
    reference_filename: Optional[str] = Form(None),
    reference_file: Optional[UploadFile] = File(None),
    reference_transcript: Optional[str] = Form(None),
    cloning_mode: str = Form("x_vector_only"),
    language: str = Form("en"),
    model_name: str = Form("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
):
    """
    Synthesize speech using Qwen3-TTS Base voice cloning and return benchmark metrics.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cloner = VoiceCloningService.get_instance()

    # Determine reference audio path
    ref_audio_path: Optional[Path] = None
    if reference_file is not None and reference_file.filename:
        clean_name = Path(reference_file.filename).name.replace(" ", "_")
        dest_path = UPLOADS_DIR / clean_name
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(reference_file.file, buffer)
        ref_audio_path = dest_path
    elif reference_filename:
        clean_name = Path(reference_filename).name
        candidate = UPLOADS_DIR / clean_name
        if candidate.is_file():
            ref_audio_path = candidate
        else:
            # Check TTS dir fallback
            fallback = Path(HERE).parent / "src" / "speech_to_speech" / "TTS" / clean_name
            if fallback.is_file():
                ref_audio_path = fallback

    if ref_audio_path is None or not ref_audio_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Reference audio is required. Please upload or select a reference voice file.",
        )

    is_xvec = (cloning_mode == "x_vector_only")
    if not is_xvec and not (reference_transcript and reference_transcript.strip()):
        raise HTTPException(
            status_code=400,
            detail="Reference transcript is required when Reference Transcript cloning mode is selected.",
        )

    try:
        result = await asyncio.to_thread(
            cloner.synthesize_voice_clone,
            text=clean_text,
            ref_audio_path=ref_audio_path,
            ref_text=reference_transcript,
            xvec_only=is_xvec,
            language=language,
            model_name=model_name,
        )
        return result.to_dict()
    except Exception as exc:
        logger.error("Voice cloning synthesis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/me")
async def me(request: Request):
    """Login state, tier, and remaining daily budget. Only meaningful in LB mode;
    sets the anonymous tracking cookie when first seen."""
    if not LIMITER_ENABLED:
        return {"enabled": False}
    tier, keys, set_cookie = auth.resolve_identity(request)
    view = auth.user_view(request, tier=tier)
    unlimited = limiter.budget_for(tier) is None
    rem = None if unlimited else await asyncio.to_thread(limiter.remaining, keys, tier)
    out = {
        "enabled": True,
        "auth": AUTH_ENABLED,
        **view,
        "remainingSec": rem,
        "limitSec": limiter.budget_for(tier),
        "loginUrl": auth.OAUTH_LOGIN_PATH if AUTH_ENABLED else None,
        "logoutUrl": auth.OAUTH_LOGOUT_PATH if AUTH_ENABLED else None,
    }
    resp = JSONResponse(out)
    if set_cookie:
        auth.set_anon_cookie(resp, set_cookie)
    return resp


@app.post("/api/search")
async def search(req: SearchRequest):
    """Proxy a Google search via Serper.dev. The key stays on the server unless
    the user brought their own (then theirs is used for this request only)."""
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")

    key = (req.key or "").strip() or SERPER_KEY
    if not key:
        # No server key and the user didn't supply one — search is unavailable.
        raise HTTPException(status_code=503, detail="Search is not configured.")

    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    payload = {"q": query, "num": MAX_RESULTS}
    try:
        async with httpx.AsyncClient(timeout=12.0) as http:
            resp = await http.post(SERPER_URL, headers=headers, json=payload)
    except httpx.RequestError as exc:
        logger.warning("Serper unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Search provider unreachable.")

    if resp.status_code != 200:
        # Serper's error body carries the real reason (e.g. "Not enough
        # credits") and contains no key, so it's safe to log and relay.
        body = resp.text[:300]
        logger.warning("Serper error %s: %s", resp.status_code, body)
        msg = None
        try:
            msg = resp.json().get("message")
        except Exception:
            pass
        detail = f"Search provider error ({resp.status_code})"
        if msg:
            detail += f": {msg}"
        raise HTTPException(status_code=502, detail=detail)

    data = resp.json()
    results = []
    for item in (data.get("organic") or [])[:MAX_RESULTS]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
            }
        )

    # A direct answer when Google has one — saves the model a hop.
    box = data.get("answerBox") or {}
    answer = box.get("answer") or box.get("snippet") or None
    if not answer:
        kg = data.get("knowledgeGraph") or {}
        answer = kg.get("description") or None

    return JSONResponse({"query": query, "answer": answer, "results": results})


@app.post("/api/calls")
async def calls(request: Request):
    """Proxy the WebRTC SDP handshake to the pinned s2s server.

    The browser can't POST /v1/realtime/calls cross-origin (the s2s server has
    no CORS middleware, and an application/sdp POST is preflighted), so it
    posts the offer here and we forward it server-side. Only the signaling hop
    goes through this proxy — the negotiated audio/data-channel media flows
    directly between the browser and the s2s server.

    Deliberately forwards ONLY to SPEECH_TO_SPEECH_URL: honouring a
    client-supplied target would make this an open proxy (SSRF). No env pin,
    no WebRTC — the client keeps such setups on the WebSocket transport."""
    if not SPEECH_TO_SPEECH_URL:
        raise HTTPException(status_code=404, detail="Not found.")

    offer = await request.body()
    url = _webrtc_calls_url(SPEECH_TO_SPEECH_URL)
    try:
        # Generous timeout: the s2s server waits for its own ICE gathering
        # (up to ~5 s) before returning the answer.
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(url, headers={"Content-Type": "application/sdp"}, content=offer)
    except httpx.RequestError as exc:
        logger.warning("s2s calls endpoint unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Speech service unreachable.")

    # Relay the answer (or the error body) as-is; keep the Location header the
    # s2s server sets on success (the call id, per the OpenAI GA contract).
    headers = {}
    if "location" in resp.headers:
        headers["Location"] = resp.headers["location"]
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/sdp"),
        headers=headers,
    )


@app.post("/api/session")
async def session(request: Request):
    """Proxy the session handshake to the load balancer, keeping its URL secret,
    and meter conversation time by tier.

    The browser POSTs here (same-origin); we resolve the caller's tier, refuse if
    today's budget is already spent (402), otherwise POST <LOAD_BALANCER_URL>/session
    and relay the JSON back. The LB body carries a per-session `connect_url`
    (compute host + short-lived token) the browser must dial directly — that one
    URL is unavoidably exposed, but the stable load-balancer address is not. On a
    successful grant we reserve the first time chunk against the day's budget."""
    if not LOAD_BALANCER_URL:
        # No LB configured — this deploy is direct-mode only; the browser should
        # never call this. 404 so it's indistinguishable from a missing route.
        raise HTTPException(status_code=404, detail="Not found.")

    login_reason = auth.oauth_login_required_reason(request)
    if login_reason:
        return _login_required_response(login_reason)

    tier, keys, set_cookie = auth.resolve_identity(request)
    # Metering runs only on the deployed Space; off-Space the LB still proxies but
    # nothing is tracked. Within metering, unlimited tiers (pro, org) aren't either.
    tracked = LIMITER_ENABLED and limiter.budget_for(tier) is not None

    # Refuse before troubling the LB if the day's budget is already gone. Done
    # here (at enqueue) so we never put a user who can't talk into the queue.
    if tracked:
        rem = await asyncio.to_thread(limiter.remaining, keys, tier)
        if rem is not None and rem <= 0:
            resp = JSONResponse(
                {"tier": tier, "reason": "limit", "remainingSec": 0}, status_code=402
            )
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    url = f"{LOAD_BALANCER_URL.rstrip('/')}/session"
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            lb = await http.post(
                url,
                headers=_load_balancer_headers(request),
                content="{}",
            )
    except httpx.RequestError as exc:
        logger.warning("Load balancer unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Speech service unreachable.")

    # The queue is full: the LB replies 503 {state:"at_capacity"}. Relay it as-is
    # so the client shows a soft "try again shortly", not a hard error.
    if lb.status_code == 503:
        body = _safe_json(lb)
        if body.get("state") == "at_capacity":
            resp = JSONResponse({"state": "at_capacity"}, status_code=503)
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    if lb.status_code == 401:
        body = _safe_json(lb)
        reason = body.get("reason", "login_required")
        if reason == "token_invalid":
            session = getattr(request, "scope", {}).get("session")
            if isinstance(session, dict):
                session.pop("oauth_info", None)
        logger.info("Session authentication rejected: %s", reason)
        return _login_required_response(reason, set_cookie)

    if lb.status_code != 200:
        # The LB's error body may name the reason (e.g. capacity); it carries no
        # secret, so relay a trimmed copy.
        logger.warning("Session handshake failed %s: %s", lb.status_code, lb.text[:300])
        raise HTTPException(status_code=502, detail=f"Session handshake failed ({lb.status_code}).")

    data = lb.json()

    # Busy pool: the LB queued us. Relay the ticket untouched — crucially with NO
    # reservation, so waiting in line never costs the day's budget.
    if data.get("state") == "queued":
        data["tier"] = tier
        resp = JSONResponse(data)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    # A slot was free: reserve the first chunk now and return the grant.
    return await _finalize_grant(data, keys, tier, tracked, set_cookie)


def _login_required_response(reason: str, set_cookie=None) -> JSONResponse:
    """Actionable 401 understood by the browser's login-required flow."""
    resp = JSONResponse(
        {
            "reason": reason,
            "loginUrl": auth.OAUTH_LOGIN_PATH if AUTH_ENABLED else None,
        },
        status_code=401,
    )
    if set_cookie:
        auth.set_anon_cookie(resp, set_cookie)
    return resp


def _load_balancer_headers(request: Request) -> dict[str, str]:
    """Headers for the server-to-server session allocation request.

    The dedicated authorization header matches the Reachy Mini client and lets
    the load balancer validate and attribute an optional HF user token without
    exposing it to browser JavaScript. Anonymous visitors send no credential.
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": LB_USER_AGENT,
    }
    token = auth.current_access_token(request)
    if token:
        headers["X-Reachy-Mini-Authorization"] = f"Bearer {token}"
    return headers


@app.get("/api/queue/{queue_id}")
async def queue_status(queue_id: str, request: Request):
    """Poll a waiting ticket: relay the position, or — when the head of the line
    claims a freed slot — reserve the budget now and return the grant. Re-checks the
    daily budget at claim, since a multi-minute wait could have spent it elsewhere."""
    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")

    login_reason = auth.oauth_login_required_reason(request)
    if login_reason:
        return _login_required_response(login_reason)

    tier, keys, set_cookie = auth.resolve_identity(request)
    tracked = LIMITER_ENABLED and limiter.budget_for(tier) is not None

    url = f"{LOAD_BALANCER_URL.rstrip('/')}/queue/{queue_id}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            lb = await http.get(url)
    except httpx.RequestError as exc:
        logger.warning("Load balancer unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Speech service unreachable.")

    if lb.status_code == 404:
        # Ticket unknown/expired (reaped after we stopped polling). Tell the client
        # to start over rather than spin.
        resp = JSONResponse({"state": "expired"}, status_code=404)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    if lb.status_code != 200:
        logger.warning("Queue poll failed %s: %s", lb.status_code, lb.text[:300])
        raise HTTPException(status_code=502, detail=f"Queue poll failed ({lb.status_code}).")

    data = lb.json()

    if data.get("state") == "queued":
        data["tier"] = tier
        resp = JSONResponse(data)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    # Claimed a slot. Re-check the budget: it may have been spent in another tab
    # during the wait. If so, refuse — the just-claimed slot is now a pending
    # session on the LB and its pending-timeout reaper reclaims it shortly.
    if tracked:
        rem = await asyncio.to_thread(limiter.remaining, keys, tier)
        if rem is not None and rem <= 0:
            resp = JSONResponse(
                {"tier": tier, "reason": "limit", "remainingSec": 0}, status_code=402
            )
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    return await _finalize_grant(data, keys, tier, tracked, set_cookie)


@app.delete("/api/queue/{queue_id}")
async def queue_leave(queue_id: str):
    """Leave the queue from the explicit 'Leave queue' button (a real fetch)."""
    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")
    await _lb_leave(queue_id)
    return {"ok": True}


@app.post("/api/queue/end")
async def queue_end(request: Request):
    """Leave the queue on teardown/tab-close (navigator.sendBeacon, which can only
    POST). Body: { queueId }. Best-effort; the LB reaps the ticket on TTL anyway."""
    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")
    qid = await _queue_id(request)
    if qid:
        await _lb_leave(qid)
    return {"ok": True}


async def _finalize_grant(data, keys, tier, tracked, set_cookie):
    """Shared grant tail (fast path or queue claim): reserve the first chunk, attach
    the metering fields the client needs, and set the anon cookie."""
    remaining = None
    if tracked and data.get("session_id"):
        await asyncio.to_thread(limiter.begin, data["session_id"], keys, tier)
        remaining = await asyncio.to_thread(limiter.remaining, keys, tier)

    data.update({
        "tier": tier,
        "limited": tracked,
        "remainingSec": remaining,
        "heartbeatSec": limiter.HEARTBEAT_SEC,
    })
    resp = JSONResponse(data)
    if set_cookie:
        auth.set_anon_cookie(resp, set_cookie)
    return resp


async def _lb_leave(queue_id: str) -> None:
    """Best-effort: tell the LB to drop a waiting ticket."""
    url = f"{LOAD_BALANCER_URL.rstrip('/')}/queue/{queue_id}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            await http.delete(url)
    except httpx.RequestError as exc:
        logger.warning("Queue leave failed: %r", exc)


def _safe_json(response) -> dict:
    try:
        body = response.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


async def _queue_id(request: Request) -> str:
    """Pull `queueId` from a JSON body, tolerating sendBeacon's blob posts."""
    try:
        data = await request.json()
    except Exception:
        return ""
    return (data or {}).get("queueId", "") if isinstance(data, dict) else ""


async def _session_id(request: Request) -> str:
    """Pull `sessionId` from a JSON body, tolerating sendBeacon's blob posts."""
    try:
        data = await request.json()
    except Exception:
        return ""
    return (data or {}).get("sessionId", "") if isinstance(data, dict) else ""


@app.post("/api/session/heartbeat")
async def session_heartbeat(request: Request):
    """Extend the live reservation one chunk at a time. `expired` once the day's
    budget is spent — the client then tears down."""
    if not LIMITER_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    sid = await _session_id(request)
    alive = bool(sid) and await asyncio.to_thread(limiter.heartbeat, sid)
    return {"expired": not alive}


@app.post("/api/session/end")
async def session_end(request: Request):
    """Clean teardown: reconcile to real elapsed time and refund the unused
    chunk. Sent via navigator.sendBeacon, so it must succeed without a response."""
    if not LIMITER_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    sid = await _session_id(request)
    if sid:
        await asyncio.to_thread(limiter.end, sid)
    return {"ok": True}


# Static front-end. Registered last so the /api routes win. `html=True` serves
# index.html at "/". The repo is public anyway, so serving the dir is fine.
app.mount("/", StaticFiles(directory=HERE, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Speech-to-Speech Demo & Voice Cloning Benchmark on http://{host}:{port}...")
    uvicorn.run("server:app", host=host, port=port, reload=False, app_dir=HERE)
