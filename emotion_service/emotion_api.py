from fastapi import FastAPI
import os
import json

app = FastAPI(title="Emotion State API")

@app.get("/current_emotion")
async def get_current_emotion():
    # Check if the file exists (in case the API starts before the camera)
    if os.path.exists("current_state.json"):
        try:
            with open("current_state.json", "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Handle the microsecond where the file is being overwritten by the camera
            return {"student_emotion": "neutral"}
    
    # Fallback if the camera hasn't written anything yet
    return {"student_emotion": "neutral"}