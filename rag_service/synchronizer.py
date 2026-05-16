import bisect
import json

class TemporalSynchronizer:
    def __init__(self, transcript_path: str):
        # Load the JSON into RAM when the server boots
        with open(transcript_path, 'r') as f:
            data = json.load(f)
            self.transcript = data.get('segments', data)

        # Create a flat, sorted list of just the start times
        self.start_times = [segment['start'] for segment in self.transcript]

    def get_transcript_segment(self, emotion_timestamp: float) -> dict:
        # Find which text chunk belongs to this exact second
        # Get the insertion point and subtract 1 to find the active interval
        index = bisect.bisect_right(self.start_times, emotion_timestamp) - 1

        # Deal with the situation when an emotion happens before anyone started
        # speaking (e.g., intro music)
        if index < 0:
            return {"text": "[Silence/No Speech Detected]"}
        
        segment = self.transcript[index]
        # Deal with the case where the emotion happened during a long pause
        # between sentences. Verify that the emotion timestamp didn't occur
        # after this specific segment ended
        if emotion_timestamp <= segment['end']:
            return segment
        else:
            return {"text": "[Silence/Pause in Speech]"}
