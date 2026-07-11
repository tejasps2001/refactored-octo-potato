#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import signal
import time
import threading

# Base Directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
EMOTION_DIR = os.path.join(ROOT_DIR, "emotion_service")
RAG_DIR = os.path.join(ROOT_DIR, "rag_service")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# Global Process List
processes = []
exit_flag = False

def get_venv_python(service_dir):
    """Detects platform and returns the appropriate virtual environment python binary path."""
    if sys.platform.startswith("win"):
        return os.path.join(service_dir, ".venv", "Scripts", "python.exe")
    return os.path.join(service_dir, ".venv", "bin", "python")

def setup_venv(service_dir, name):
    """Enforces virtual environment initialization and syncs dependencies."""
    venv_dir = os.path.join(service_dir, ".venv")
    python_bin = get_venv_python(service_dir)
    
    if not os.path.exists(venv_dir):
        print(f"[{name}] Creating virtual environment in {venv_dir}...", flush=True)
        # Use current system python interpreter to create the venv
        subprocess.run([sys.executable, "-m", "venv", ".venv"], cwd=service_dir, check=True)
    
    req_file = os.path.join(service_dir, "requirements.txt")
    if os.path.exists(req_file):
        print(f"[{name}] Inspecting environment boundaries and syncing requirements...", flush=True)
        # Upgrade pip quietly
        subprocess.run([python_bin, "-m", "pip", "install", "--upgrade", "pip"], 
                       cwd=service_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Install/update packages
        subprocess.run([python_bin, "-m", "pip", "install", "-r", "requirements.txt"], 
                       cwd=service_dir, check=True)

def log_pipe(stream, prefix):
    """Pipes stdout/stderr of subprocesses to main console with clear prefixes."""
    try:
        for line in iter(stream.readline, ''):
            if not line:
                break
            print(f"{prefix} {line.strip()}", flush=True)
    except Exception:
        pass

def start_log_thread(process, name):
    """Starts background threads to stream output lines in real-time."""
    t_out = threading.Thread(target=log_pipe, args=(process.stdout, f"[{name}]"), daemon=True)
    t_err = threading.Thread(target=log_pipe, args=(process.stderr, f"[{name}_ERR]"), daemon=True)
    t_out.start()
    t_err.start()

def cleanup(signum=None, frame=None):
    """Safe teardown guardrail to terminate and wait on all child processes."""
    global exit_flag
    if exit_flag:
        return
    exit_flag = True
    
    print("\n[SYSTEM] Initiating teardown & cleaning up ports...", flush=True)
    
    # 1. Terminate all active processes
    for p, name in processes:
        if p.poll() is None:
            print(f"[SYSTEM] Terminating {name} (PID: {p.pid})...", flush=True)
            p.terminate()
            
    # 2. Wait or Kill if unresponsive
    for p, name in processes:
        try:
            p.wait(timeout=5)
            print(f"[SYSTEM] {name} (PID: {p.pid}) terminated cleanly.", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[SYSTEM] {name} (PID: {p.pid}) did not exit. Force killing...", flush=True)
            p.kill()
            p.wait()
            print(f"[SYSTEM] {name} (PID: {p.pid}) killed.", flush=True)
            
    print("[SYSTEM] Workspace is cleared and ports are opened.", flush=True)
    if signum is not None:
        sys.exit(0)

# Register Signal Handlers
if hasattr(signal, "SIGINT"):
    signal.signal(signal.SIGINT, cleanup)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, cleanup)

def validate_knowledge_base():
    """Verifies whether the 'rag_service/data/' folder exists and contains files."""
    data_dir = os.path.join(RAG_DIR, "data")
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir) or not any(os.path.isfile(os.path.join(data_dir, f)) for f in os.listdir(data_dir)):
        print("[ERROR] Knowledge base empty. Please place lecture files (PDF/MP4) inside rag_service/data/ before running the pipeline.")
        sys.exit(1)

def verify_ollama_and_models():
    """Verifies that Ollama is in PATH and the required models are pulled."""
    import shutil
    if shutil.which("ollama") is None:
        print("[ERROR] The system requires an active Ollama environment installation. Please download and install Ollama from https://ollama.com before running the pipeline.")
        sys.exit(1)

    required_models = ["gemma3:4b", "nomic-embed-text"]
    try:
        # Run 'ollama list' to get the current downloaded models
        result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        installed_models = result.stdout
    except Exception as e:
        print(f"[ERROR] Failed to run 'ollama list': {e}")
        sys.exit(1)

    for model in required_models:
        if model not in installed_models:
            print(f"[SYSTEM] Model {model} not found locally. Initiating automated pull via Ollama...")
            try:
                # Stream the pull CLI output directly to the terminal
                subprocess.run(["ollama", "pull", model], check=True)
            except Exception as e:
                print(f"[ERROR] Failed to pull model {model}: {e}")
                sys.exit(1)

def main():
    # Validate knowledge base at the entry point of the pipeline
    validate_knowledge_base()

    # Verify Ollama installation and download required model weights
    verify_ollama_and_models()

    parser = argparse.ArgumentParser(
        description="Unified Cross-Platform Orchestrator for RAG & Emotion Telemetry Services"
    )
    parser.add_argument("-e", "--emotion-api", action="store_true", 
                        help="Launch the FastAPI Emotion State Backend (Port 8001)")
    parser.add_argument("-c", "--camera", action="store_true", 
                        help="Launch the MediaPipe Webcam Tracking Loop")
    parser.add_argument("-r", "--rag", action="store_true", 
                        help="Launch the FastAPI Core RAG Backend Pipeline (Port 8000)")
    parser.add_argument("-f", "--frontend", "--ui", dest="ui", action="store_true", 
                        help="Launch the Streamlit User Interface View (Port 8501)")
    parser.add_argument("--backend", action="store_true", 
                        help="Launch both FastAPI backends (RAG and Emotion)")
    
    args = parser.parse_args()
    
    # If no flags are passed, run everything by default
    run_emotion = args.emotion_api or args.backend
    run_rag = args.rag or args.backend
    run_camera = args.camera
    run_ui = args.ui
    
    if not (args.emotion_api or args.camera or args.rag or args.ui or args.backend):
        run_emotion = True
        run_rag = True
        run_camera = True
        run_ui = True

    # Build environment to ensure output is unbuffered
    sub_env = os.environ.copy()
    sub_env["PYTHONUNBUFFERED"] = "1"
    # Force subprocesses to use UTF-8 for their stdio, fixing Unicode errors on Windows
    sub_env["PYTHONIOENCODING"] = "utf-8"

    try:
        # Step 1: Environment Sync / Checks
        if run_emotion or run_camera:
            setup_venv(EMOTION_DIR, "EMOTION_SERVICE")
        if run_rag:
            setup_venv(RAG_DIR, "RAG_SERVICE")
        if run_ui:
            setup_venv(FRONTEND_DIR, "FRONTEND")

        # Step 2: Start Emotion API
        if run_emotion:
            venv_python = get_venv_python(EMOTION_DIR)
            cmd = [venv_python, "-m", "uvicorn", "emotion_api:app", "--host", "0.0.0.0", "--port", "8001"]
            print(f"[SYSTEM] Starting EMOTION_API...", flush=True)
            p = subprocess.Popen(cmd, cwd=EMOTION_DIR, env=sub_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, bufsize=1, errors='replace', encoding='utf-8')
            processes.append((p, "EMOTION_API"))
            print(f"[SYSTEM] Background process active: EMOTION_API (PID: {p.pid})", flush=True)
            start_log_thread(p, "EMOTION_API")

        # Step 3: Start Camera Loop
        if run_camera:
            venv_python = get_venv_python(EMOTION_DIR)
            cmd = [venv_python, "run_camera.py"]
            print(f"[SYSTEM] Starting CAMERA_LOOP...", flush=True)
            p = subprocess.Popen(cmd, cwd=EMOTION_DIR, env=sub_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, bufsize=1, errors='replace', encoding='utf-8')
            processes.append((p, "CAMERA_LOOP"))
            print(f"[SYSTEM] Background process active: CAMERA_LOOP (PID: {p.pid})", flush=True)
            start_log_thread(p, "CAMERA_LOOP")

        # Step 4: Start RAG Service
        if run_rag:
            venv_python = get_venv_python(RAG_DIR)
            
            # Run Ingestion synchronously first
            print("[SYSTEM] Running automated ingestion pipeline...", flush=True)
            subprocess.run([venv_python, "ingest.py"], cwd=RAG_DIR, env=sub_env, check=True)
            
            cmd = [venv_python, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
            print(f"[SYSTEM] Starting RAG_SERVICE...", flush=True)
            p = subprocess.Popen(cmd, cwd=RAG_DIR, env=sub_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, bufsize=1, errors='replace', encoding='utf-8')
            processes.append((p, "RAG_SERVICE"))
            print(f"[SYSTEM] Background process active: RAG_SERVICE (PID: {p.pid})", flush=True)
            start_log_thread(p, "RAG_SERVICE")

        # Step 5: Start Frontend
        if run_ui:
            venv_python = get_venv_python(FRONTEND_DIR)
            cmd = [venv_python, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
            print(f"[SYSTEM] Starting STREAMLIT_UI...", flush=True)
            p = subprocess.Popen(cmd, cwd=FRONTEND_DIR, env=sub_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, bufsize=1, errors='replace', encoding='utf-8')
            processes.append((p, "STREAMLIT_UI"))
            print(f"[SYSTEM] Foreground process active: STREAMLIT_UI (PID: {p.pid})", flush=True)
            start_log_thread(p, "STREAMLIT_UI")

        # Keep main thread alive and monitor children
        print("[SYSTEM] All requested services launched. Listening for terminal events...", flush=True)
        while True:
            for p, name in processes:
                ret = p.poll()
                if ret is not None:
                    print(f"[SYSTEM] Critical failure: {name} (PID: {p.pid}) exited with code {ret}.", flush=True)
                    raise SystemExit(ret)
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        sys.exit(e.code)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
