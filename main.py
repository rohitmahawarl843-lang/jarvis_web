"""
Jarvis AI — Web/Server Version
Browser se chalta hai, kisi bhi domain par deploy ho sakta hai.
(Windows PC-control features iss version mein NAHI hain — server
sirf apna khud ka OS control kar sakta hai, kisi user ke ghar ke
PC ka nahi.)
"""
import os
import io
import json
import uuid
import mimetypes

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from google import genai
from google.genai import types

APP_DIR = os.path.dirname(os.path.abspath(__file__))

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")  # fast/low-latency
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "20"))

SYSTEM_PROMPT = """You are Jarvis, a friendly AI assistant.
You speak in Hinglish naturally. Be helpful and concise.
Agar user koi file (image, video, PDF, spreadsheet, document) attach kare,
uska data dhyan se dekho/padho aur uske sawaal ka sahi jawab do ya data ka
summary/analysis do."""

client = genai.Client(api_key=API_KEY) if API_KEY else None

# Simple in-memory per-browser-session chat history.
# NOTE: yeh server restart hone par (ya scale-out/multiple instances par)
# khatam ho jaata hai. Production ke liye Redis/DB use karo.
sessions = {}

# Mime types jo Gemini seedha (inline) samajh leta hai
INLINE_PREFIXES = ("image/", "video/", "audio/")
INLINE_EXACT = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "text/html",
    "text/xml",
    "text/rtf",
    "text/markdown",
}


def build_file_part(filename: str, content_type: str, data: bytes):
    """Uploaded file ko Gemini ke liye sahi Part mein convert karta hai."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    mime = (content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream").split(";")[0].strip()

    # Excel spreadsheets -> CSV text mein convert karo (Gemini xlsx seedha nahi leta)
    if ext in ("xlsx", "xls"):
        try:
            import pandas as pd
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
            chunks = []
            for name, df in sheets.items():
                chunks.append(f"--- Sheet: {name} ---\n{df.to_csv(index=False)}")
            csv_text = "\n\n".join(chunks)
            return types.Part.from_text(text=f"[Uploaded spreadsheet: {filename}]\n{csv_text}")
        except Exception as e:
            return types.Part.from_text(text=f"[Spreadsheet '{filename}' padhne mein error: {e}]")

    # Word documents -> text extract karo
    if ext == "docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
            return types.Part.from_text(text=f"[Uploaded document: {filename}]\n{text}")
        except Exception as e:
            return types.Part.from_text(text=f"[Document '{filename}' padhne mein error: {e}]")

    # Images, video, audio, pdf, plain-text formats — Gemini inline le leta hai
    if mime.startswith(INLINE_PREFIXES) or mime in INLINE_EXACT:
        return types.Part.from_bytes(data=data, mime_type=mime)

    # Fallback: text ke roop mein decode karne ki koshish
    try:
        text = data.decode("utf-8", errors="ignore")
        return types.Part.from_text(text=f"[Uploaded file: {filename}]\n{text}")
    except Exception:
        return types.Part.from_text(text=f"[File '{filename}' (type: {mime}) is format mein analyze nahi ho saka]")


def get_chat_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=800,
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
async def chat(
    message: str = Form(""),
    session_id: str = Form("default"),
    file: UploadFile = File(None),
):
    text = (message or "").strip()

    if not client:
        def err_stream():
            yield f"data: {json.dumps({'error': 'Server par GEMINI_API_KEY set nahi hai.'})}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    has_file = file is not None and bool(file.filename)

    if not text and not has_file:
        def empty_stream():
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    chat_session = get_chat_session(session_id)

    parts = []
    if has_file:
        data = await file.read()
        parts.append(build_file_part(file.filename, file.content_type, data))

    parts.append(types.Part.from_text(
        text=text or "Is attached file ka data dhyan se dekho aur uska summary/analysis do."
    ))

    def event_stream():
        try:
            for chunk in chat_session.send_message_stream(parts):
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
