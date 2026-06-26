# Research Log - Emotion-Aware RAG Tutor System

## Milestone: Phase 2 (Knowledge & Pedagogy Control)

### 1. Architectural Mutated Files
- **[api.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/rag_service/api.py)**:
  - Transitioned the LLM to `gemma3:4b` from `gemma3:270m` to resolve instruction-following failures.
  - Implemented an LCEL `RunnableParallel` pipeline integrating a `RunnableLambda` to fetch live student emotions dynamically from `http://localhost:8001/current_emotion` with a 1.0-second timeout.
  - Restructured prompt engineering to guarantee strict syllabus boundaries and dynamic pedagogical tone adaptation.
  - Implemented verbose terminal logging to display pipeline inputs: `[Question, Retrieved Context Chunks Count, Injected Live Emotion]`.
- **[session_logger.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/rag_service/session_logger.py)**:
  - Fixed syntax, schema mismatches (e.g. `video_timestamp` column naming), and parameter mismatches inside the `log_engagement` SQLite execution method.
- **[run_rag.sh](file:///home/tejasps/Documents/AI/refactored-octo-potato/run_rag.sh)**:
  - Updated usage instructions and verified option switches for decoupled microservice orchestration.
- **[run.sh](file:///home/tejasps/Documents/AI/refactored-octo-potato/emotion_service/run.sh)**:
  - Deleted the redundant bootstrap script to consolidate processes.

### 2. The LLM Target Swap
- **Baseline Model (`gemma3:270m`)**:
  - This model demonstrated a severe instruction-following threshold failure. When instructions regarding context boundaries and tone adaptation were given, the model fell back to the negative clause ("This topic was not covered in the lecture material.") even when the information was explicitly provided in the retrieval context.
- **Upgraded Model (`gemma3:4b`)**:
  - This model demonstrated clearer understanding of the separation between system instructions and retrieval context. It correctly responded using the provided context when relevant, and strictly triggered the fallback statement when queried on topics outside the context.

### 3. The Prompt Blueprint
```
You are a helpful teaching assistant. You must answer the student's question ONLY using the provided retrieved context from the lecture transcript and notes. Do not use any outside knowledge. If the answer to the question cannot be found or inferred from the provided context, you must output exactly: "This topic was not covered in the lecture material." and nothing else.

CRITICAL INSTRUCTION: The camera detects that the student is currently feeling: {student_emotion}.
Adapt your pedagogical tone accordingly:
- If they are feeling "Frustration" or "Confusion" (or related negative/struggling emotions), be extra patient, break down the steps clearly, offer encouragement, and guide them step-by-step.
- If they are feeling "Engaged" or "Concentration" or "Joy", provide a concise, direct, and technical answer.
- For other emotional states (like "Neutral", "Bored", "Note-Taking"), maintain a balanced, supportive, and clear explanation.

Context:
{context}

Question: {question}

Answer:
```

### 4. Verification Trace
- **Active Emotion Service (Injected Live Emotion: Frustration)**:
  - Request: `curl -X POST http://localhost:8000/chat -d '{"question": "What is the time complexity of Bubble Sort?", "student_emotion": "Neutral"}'`
  - Output: `{"answer":"Bubble Sort has a worst-case time complexity of O(n squared). It works by repeatedly swapping adjacent elements if they are in the wrong order."}`
  - Trace Log:
    ```
    --- DEBUG: Received Emotion: Neutral ---
    [DEBUG] Pipeline Inputs:
      - Question: What is the time complexity of Bubble Sort?
      - Retrieved Context Chunks Count: 1
      - Injected Live Emotion: Frustration
    ```
- **Unreachable Emotion Service (Fallback to neutral)**:
  - Request: `curl -X POST http://localhost:8000/chat -d '{"question": "What is the time complexity of Bubble Sort?", "student_emotion": "neutral"}'`
  - Output: `{"answer":"Bubble Sort has a worst-case time complexity of O(n squared)."}`
  - Trace Log:
    ```
    --- DEBUG: Received Emotion: neutral ---
    [DEBUG] Pipeline Inputs:
      - Question: What is the time complexity of Bubble Sort?
      - Retrieved Context Chunks Count: 1
      - Injected Live Emotion: neutral
    ```
- **Out-of-Syllabus / Missing Context**:
  - Request: `curl -X POST http://localhost:8000/chat -d '{"question": "What is Quick Sort?", "student_emotion": "neutral"}'`
  - Output: `{"answer":"This topic was not covered in the lecture material."}`
  - Trace Log:
    ```
    --- DEBUG: Received Emotion: neutral ---
    [DEBUG] Pipeline Inputs:
      - Question: What is Quick Sort?
      - Retrieved Context Chunks Count: 1
      - Injected Live Emotion: neutral
    ```

## Milestone: Phase 3 (Struggle Tracking & Active Research Core)

### 1. Architectural Mutated Files
- **[session_logger.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/rag_service/session_logger.py)**:
  - Extended SQLite schema to include tables: `video_navigation_logs`, `chat_logs`, `struggle_logs`, and `post_video_questions` with a mandatory `session_id TEXT` column to isolate runs.
  - Formulated an asynchronous queue-based sequential write worker utilizing a background daemon thread to prevent database lockups and collisions.
  - Built getters to extract engagement spikes, navigation rewinds, chat history, and generated questions.
- **[api.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/rag_service/api.py)**:
  - Created validation schemas for log navigation and QA generation requests.
  - Implemented `@app.post("/log_navigation")` endpoint to receive scrubbing events and resolve them against the synchronizer.
  - Modified `@app.post("/chat")` endpoint to save conversations to the chat history database.
  - Formulated the `aggregate_struggles` helper to pull raw database logs and compile them into a unified Struggle Log.
  - Built `@app.post("/generate_qa")` endpoint to parse struggles, extract vector contexts, prompt the LLM, parse responses using a robust regex/JSON fallback extractor, and log the generated questions as serialized JSON.

### 2. The Prompt Blueprint
```
System: You are an expert AI Research Assistant. Your task is to analyze the student's Struggle Log and generate 3 custom, high-quality, concept-check test questions. 
Target the specific topics or transcript segments where the student struggled (indicated by emotional spikes, video rewinds, or chat questions).
You must use ONLY the provided Context notes to ensure correct facts.
Format your output exactly as a JSON list of 3 strings, and nothing else.

Context Notes:
{context}

Struggle Log:
{struggle_log}

Questions (JSON array of 3 strings):
```

### 3. Verification Trace
- **POST `/generate_qa`**:
  - Request: `curl -X POST http://localhost:8000/generate_qa -d '{"session_id": "session_alpha"}'`
  - Output:
    ```json
    {
      "questions": [
        "What is the worst-case time complexity of Bubble Sort, and why does it typically have a higher complexity compared to other sorting algorithms?",
        "Merge Sort utilizes a divide-and-conquer strategy and achieves a time complexity of O(n log n). Explain briefly what this 'divide-and-conquer' approach entails in the context of Merge Sort.",
        "Binary search demands a sorted array to function correctly. Describe the time complexity of Binary search and why its performance is dependent on the input array being sorted."
      ]
    }
    ```
  - Trace Log:
    ```
    [DEBUG] [Aggregated Struggles Vector Count]: 3
    [DEBUG] [Retrieved Vector Space Chunks]: 1
    [DEBUG] [LLM Payload Handoff]: ...
    ```

## Milestone: Phase 4 (Data Validation, Feedback Loop, & Calibration)

### 1. Architectural Mutated Files
- **[emotion_recognizer_setup.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/emotion_service/core/emotion_recognizer_setup.py)**:
  - Guarded `calibrate_scores` against divisions by zero during dynamic 10-second calibration.
  - Added a structured Python dictionary that broadcasts the student's current emotion, calibration progress, and raw facial data at one-second intervals.
  - Modified `record_result` to log raw key blendshape scores at 1-second intervals.
- **[emotion_api.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/emotion_service/emotion_api.py)**:
  - Modified `/current_emotion` GET endpoint to load and expose the complete metrics.
  - Integrated console trace prints showing payload dispatches.
- **[api.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/rag_service/api.py)**:
  - Modified `fetch_live_emotion` to print microservice bridge traces.
  - Implemented calibration state routing fallback, mapping the `"Calibrating..."` emotion state to `"Neutral"` in the prompt payload to prevent LLM stutter.
  - Integrated explicit inline LCEL payload prints right before LLM handoff.

### 2. Verification Trace
- **Emotion API Backend Trace**:
  ```
  [DEBUG] [Bridge] Emotion payload dispatched: {'student_emotion': 'Calibrating...', 'calibration_progress': 0.45, 'raw_scores': {'browDownLeft': 0.3}}
  INFO:     127.0.0.1:60254 - "GET /current_emotion HTTP/1.1" 200 OK
  ```
- **RAG Backend Telemetry Trace**:
  ```
  [DEBUG] [Bridge] Polled current emotion from service: Calibrating...
  [DEBUG] Pipeline Inputs:
    - Question: What is the worst-case complexity of Bubble Sort?
    - Retrieved Context Chunks Count: 1
    - Injected Live Emotion: Calibrating...
  [DEBUG] [LCEL Chain] Bound prompt payload state: {'context': 'Bubble Sort has a worst-case time complexity of O(n squared). It works by repeatedly swapping adjacent elements if they are in the wrong order.\n\nMerge Sort is a stable, comparison-based sorting algorithm with a time complexity of O(n log n) in all cases. It uses a divide-and-conquer strategy.\n\nBinary search requires a sorted array and runs in O(log n) time complexity.', 'question': 'What is the worst-case complexity of Bubble Sort?', 'student_emotion': 'Neutral'}
  INFO:     127.0.0.1:49064 - "POST /chat HTTP/1.1" 200 OK
  ```

## Milestone: Phase 5 (Unified User Interface Integration)

### 1. Architectural Mutated Files
- **[app.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/frontend/app.py)**:
  - Formulated a two-column Streamlit dashboard layout containing an embedded HTML5 video player and a chat assistant.
  - Implemented secure, throttled, and debounced window postMessage events within the player iframe.
  - Built a parent-side web messaging listener that handles asynchronous metrics forwarding, SQLite DB logging, and hidden state variable propagation without blocking input draw loops.
  - Created a researcher telemetry dashboard tracking Active Session ID, Current Playhead Position, Extracted Emotion State, and Queue Latency Metric.
  - Added a dynamic calibration progress bar reading from local files.
  - Formulated the bottom-row assessment block to invoke post-video check questions.

### 2. Operational Layout & Decoupling
- Streamlit remains completely stateless and runs isolated from subprocess camera thread initialization.
- Camera vision loop and API services are executed separately in the background from the root bash script.
- Cross-origin issues are avoided by using iframe message passing instead of direct raw AJAX calls to backend services inside the video component.

## Milestone: Ingestion - Automated Multimedia Preprocessing

### 1. Architectural Mutated Files
- **[ingest.py](file:///home/tejasps/Documents/AI/refactored-octo-potato/rag_service/ingest.py)**:
  - Extended multi-source discovery loop to accept `.mp3`, `.wav`, `.mp4`, `.mkv` files.
  - Implemented an `ffmpeg` system dependency validation check using `which ffmpeg`.
  - Added a deterministic selection routine prioritising video over audio, preventing database collisions.
  - Implemented a caching mechanism matching the `filename` metadata in `transcript.json` to skip redundant Whisper model loads.
  - Integrated subprocess-bound audio track extraction using `ffmpeg` with silent logs configuration.
  - Programmed transcription using `faster-whisper` (base size on CPU with INT8 quantization).
  - Integrated execution performance trace prints: `[DEBUG] [Ingestion Pipeline] Track Isolation Time: XXs | Whisper Processing Time: XXs`.
  - Configured automatic RAG indexing to load the synthesized JSON transcript and commit document chunks to ChromaDB.
- **[run.sh](file:///home/tejasps/Documents/AI/refactored-octo-potato/run.sh)**:
  - Integrated automatic execution of the ingestion pipeline before launching uvicorn in the RAG startup block.

### 2. Verification Trace
- **First-run Execution (transcribing media file)**:
  ```
  Starting multimedia preprocessing for: ./data/SQL 1 Clip 1 [8Uxt9scWJBY].mkv
  Initializing local faster-whisper base model (CPU, INT8)...
  [DEBUG] [Ingestion Pipeline] Track Isolation Time: 0.67s | Whisper Processing Time: 48.25s
  Transcription complete. Output saved to transcript.json.
  Starting ingestion process...
  Loading: ./data/lecture_notes.txt
  Loading: ./data/SQL Part 1 - Basic Queries - Database Systems.pdf
  Loading generated transcript: ./data/transcript.json
  Embedding and saving to ChromaDB...
  Ingestion complete. Database is ready.
  ```
- **Cached Execution (skipping Whisper processing)**:
  ```
  [DEBUG] [Ingestion Pipeline] Transcript for SQL 1 Clip 1 [8Uxt9scWJBY].mkv already exists. Skipping Whisper processing.
  Starting ingestion process...
  Loading: ./data/lecture_notes.txt
  Loading: ./data/SQL Part 1 - Basic Queries - Database Systems.pdf
  Loading generated transcript: ./data/transcript.json
  Embedding and saving to ChromaDB...
  Ingestion complete. Database is ready.
  ```


