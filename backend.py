from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from fastapi import File, UploadFile
from Backend.transcript_generator import extract_audio, transcribe
from Backend.script_generator import fix_sentence
from Backend.translator import translate_sentence

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    video_path = f"temp_{video_id}.mp4"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    audio_path = f"temp_{video_id}.wav"
    try:
        extract_audio(video_path, audio_path)
        text = transcribe(audio_path)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return {"transcription": text}

@app.post("/seg_pun_restoration")
async def script_restore(text: str = Form(...)):
    script = fix_sentence(text)
    return {"en_script": script}

@app.post("/translate")
async def translate(text: str = Form(...)):
    script = translate_sentence(text)
    return {"sin_script": script}