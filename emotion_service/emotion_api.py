from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json

app = FastAPI(title="Emotion State API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/current_emotion")
async def get_current_emotion():
    payload = {"student_emotion": "Neutral", "calibration_progress": 0.0, "raw_scores": {}}
    if os.path.exists("current_state.json"):
        try:
            with open("current_state.json", "r") as f:
                payload = json.load(f)
        except Exception:
            pass
            
    print(f"[DEBUG] [Bridge] Emotion payload dispatched: {payload}")
    return payload