"""
Jarvis AI — Web/Server Version
Browser se chalta hai, kisi bhi domain par deploy ho sakta hai.
(Windows PC-control features iss version mein NAHI hain — server
sirf apna khud ka OS control kar sakta hai, kisi user ke ghar ke
PC ka nahi.)
"""
import os
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from google import genai
from google.genai import types

APP_DIR = os.path.dirname(os.path.abspath(__file__))

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")  # fast/low-latency
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "20"))

SYSTEM_PROMPT = """You are Jarvis, a friendly AI assistant.
You speak in Hinglish naturally. Be helpful and concise."""

client = genai.Client(api_key=API_KEY) if API_KEY else None

# Simple in-memory per-browser-session chat history.
# NOTE: yeh server restart hone par (ya scale-out/multiple instances par)
# khatam ho jaata hai. Production ke liye Redis/DB use karo.
sessions = {}


def get_chat_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=400,
                temperature=0.7,
            ),
        )
    return sessions[session_id]


app = FastAPI(title="Jarvis AI Web")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "api_key_configured": bool(API_KEY)}


@app.post("/api/session")
def new_session():
    return {"session_id": str(uuid.uuid4())}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    text = (body.get("message") or "").strip()
    session_id = body.get("session_id") or "default"

    if not client:
        def err_stream():
            yield f"data: {json.dumps({'error': 'Server par GEMINI_API_KEY set nahi hai.'})}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    if not text:
        def empty_stream():
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    chat_session = get_chat_session(session_id)

    def event_stream():
        try:
            for chunk in chat_session.send_message_stream(text):
                if chunk.text:
                    yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx buffering off — streaming fast rahe
        },
    )


@app.post("/api/clear")
async def clear(request: Request):
    body = await request.json()
    session_id = body.get("session_id") or "default"
    sessions.pop(session_id, None)
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
