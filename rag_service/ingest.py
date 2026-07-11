import os
import glob
import time
import json
import shutil
import subprocess
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from video_utils import find_media_file
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from faster_whisper import WhisperModel

DB_DIR = "./chroma_db"
DATA_DIR = "./data"

def get_loader(file_path):
    """Factory function to route files to the correct parser based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    else:
        return None

def preprocess_multimedia():
    """Finds media files, extracts audio, and transcribes them using faster-whisper."""
    selected_file = find_media_file(DATA_DIR)
    if not selected_file:
        return

    basename = os.path.basename(selected_file)
    transcript_path = os.path.join(DATA_DIR, "transcript.json")

    # Transcription Caching Check
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r") as f:
                existing_data = json.load(f)
            if existing_data.get("metadata", {}).get("filename") == basename:
                print(f"[DEBUG] [Ingestion Pipeline] Transcript for {basename} already exists. Skipping Whisper processing.")
                return
        except Exception:
            pass

    # System Dependencies Check
    if shutil.which("ffmpeg") is None:
        print("[ERROR] [Ingestion Pipeline] ffmpeg binary not found in path. Skipping multimedia ingestion.")
        return

    print(f"Starting multimedia preprocessing for: {selected_file}")
    
    # Process audio track isolation
    ext = os.path.splitext(selected_file)[1].lower()
    audio_path = selected_file
    temp_audio = None
    isolation_time = 0.0

    if ext in [".mp4", ".mkv"]:
        start_iso = time.time()
        temp_dir = tempfile.gettempdir()
        temp_audio = os.path.join(temp_dir, f"isolated_{int(time.time())}.wav")
        # ffmpeg with -loglevel error -y option supresses logs
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", selected_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio]
        try:
            subprocess.run(cmd, check=True)
            audio_path = temp_audio
            isolation_time = time.time() - start_iso
        except Exception as e:
            print(f"[ERROR] [Ingestion Pipeline] Audio isolation failed: {e}")
            return

    # Run Transcription
    start_whisper = time.time()
    print("Initializing local faster-whisper base model (CPU, INT8)...")
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        transcript_data = {
            "metadata": {
                "filename": basename,
                "language": info.language
            },
            "segments": []
        }
        
        segment_id = 1
        for segment in segments:
            transcript_data["segments"].append({
                "id": segment_id,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            segment_id += 1
            
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
        whisper_time = time.time() - start_whisper
        print(f"[DEBUG] [Ingestion Pipeline] Track Isolation Time: {isolation_time:.2f}s | Whisper Processing Time: {whisper_time:.2f}s")
        print("Transcription complete. Output saved to transcript.json.")

    except Exception as e:
        print(f"[ERROR] [Ingestion Pipeline] Whisper transcription failed: {e}")

    finally:
        # Clean up temp file
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass

def ingest_documents():
    # Automatically run multimedia transcription pipeline first
    preprocess_multimedia()

    print("Starting ingestion process...")
    all_files = glob.glob(os.path.join(DATA_DIR, "*.*"))
    
    docs = []
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        # Skip raw media binary files
        if ext in {".mp3", ".wav", ".mp4", ".mkv"}:
            continue
            
        loader = get_loader(file_path)
        if loader:
            print(f"Loading: {file_path}")
            docs.extend(loader.load())

    # Load transcript.json if it exists
    transcript_path = os.path.join(DATA_DIR, "transcript.json")
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r") as f:
                data = json.load(f)
            segments = data.get("segments", [])
            if segments:
                full_text = " ".join(seg["text"] for seg in segments)
                print(f"Loading generated transcript: {transcript_path}")
                docs.append(Document(page_content=full_text, metadata={"source": "transcript.json"}))
        except Exception as e:
            print(f"Error loading transcript into vector indexing: {e}")

    if not docs:
        print("No readable documents found in the data directory.")
        return

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # Embed and persist to disk
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Embedding and saving to ChromaDB...")
    
    Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print("Ingestion complete. Database is ready.")

if __name__ == "__main__":
    ingest_documents()