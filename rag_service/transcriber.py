import json
import os
from faster_whisper import WhisperModel

def transcribe_audio(file_path: str, output_path: str, model_size: str = "medium", compute_type: str = "int8"):
    """
    Transcribes an audio/video file and saves a time-stamped JSON transcript.
    """
    print(f"Initializing Whisper model '{model_size}' on CPU with {compute_type} quantization...")
    
    # Initialize the model 
    # device="cpu" ensures it doesn't crash looking for a GPU
    # compute_type="int8" is the critical flag that compresses the math to fit your RAM
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    print(f"Starting transcription for: {file_path}")
    
    # The model.transcribe() function returns a generator of segments.
    segments, info = model.transcribe(file_path, beam_size=5)

    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    # Prepare the JSON structure
    transcript_data = {
        "metadata": {
            "filename": os.path.basename(file_path),
            "model": model_size,
            "compute_type": compute_type,
            "language": info.language
        },
        "segments": []
    }

    # Iterate through the generator and extract the exact temporal data
    segment_id = 1
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        # Append the structured object to our array
        transcript_data["segments"].append({
            "id": segment_id,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })
        segment_id += 1

    # Save the array to the hard drive
    print(f"\nTranscription complete. Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

# --- Execution Block ---
if __name__ == "__main__":
    # Define where your input files live and where the JSON should go
    input_file = "./data/lecture_01.mp3" # Update this to match your actual file path
    output_json = "./data/transcript.json"
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Check if the input file actually exists before running
    if not os.path.exists(input_file):
         print(f"ERROR: Could not find input file at {input_file}")
    else:
         transcribe_audio(input_file, output_json)