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
