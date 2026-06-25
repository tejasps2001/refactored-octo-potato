from fastapi import FastAPI
import os
import json

app = FastAPI(title="Emotion State API")

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