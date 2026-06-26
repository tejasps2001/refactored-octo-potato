from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import requests
import re
import json

from session_logger import SessionLogger
from synchronizer import TemporalSynchronizer

# 1. Initialize FastAPI
app = FastAPI(title="Lecture Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/video")
def get_video():
    video_path = os.path.join(os.path.dirname(__file__), "data", "lecture.mp4")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": "Video file not found"}

TRANSCRIPT_PATH = os.path.join(os.path.dirname(__file__), "data",
                               "transcript.json")

try:
    synchronizer = TemporalSynchronizer(TRANSCRIPT_PATH)
    db_logger = SessionLogger()
    print("Synchronizer and Session Logger initialized successfully.")
except Exception as e:
    print(f"Initialization Error: {e}")

# Pydantic Models for Data Validation
class EmotionLogRequest(BaseModel):
    session_id: str = "default_session"
    video_timestamp: float
    emotion_state: str

class ChatRequest(BaseModel):
    session_id: str = "default_session"
    question: str
    student_emotion: str | None = "neutral"

class NavigationLogRequest(BaseModel):
    session_id: str = "default_session"
    timestamp_from: float
    timestamp_to: float
    event_type: str

class QAGenerationRequest(BaseModel):
    session_id: str = "default_session"

# Setup the LangChain Components globally
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOllama(
    model="gemma3:4b", 
    temperature=0.0,
    num_predict=256,
    num_ctx=2048,
    num_thread=4
)

template = """You are a helpful teaching assistant. You must answer the student's question ONLY using the provided retrieved context from the lecture transcript and notes. Do not use any outside knowledge. If the answer to the question cannot be found or inferred from the provided context, you must output exactly: "This topic was not covered in the lecture material." and nothing else.

CRITICAL INSTRUCTION: The camera detects that the student is currently feeling: {student_emotion}.
Adapt your pedagogical tone accordingly:
- If they are feeling "Frustration" or "Confusion" (or related negative/struggling emotions), be extra patient, break down the steps clearly, offer encouragement, and guide them step-by-step.
- If they are feeling "Engaged" or "Concentration" or "Joy", provide a concise, direct, and technical answer.
- For other emotional states (like "Neutral", "Bored", "Note-Taking"), maintain a balanced, supportive, and clear explanation.

Context:
{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def fetch_live_emotion(inputs):
    try:
        response = requests.get("http://localhost:8001/current_emotion", timeout=1.0)
        if response.status_code == 200:
            result_emotion = response.json().get("student_emotion", "Neutral")
            print(f"[DEBUG] [Bridge] Polled current emotion from service: {result_emotion}")
            return result_emotion
    except Exception:
        print("[DEBUG] [Bridge] Polling failed or timed out. Defaulting to Neutral.")
    return inputs.get("student_emotion", "Neutral") or "Neutral"

def retrieve_docs(inputs):
    return retriever.invoke(inputs["question"])

def print_telemetry_and_format(inputs):
    question = inputs["question"]
    docs = inputs["context_docs"]
    emotion = inputs["student_emotion"]
    
    # Route "Neutral" fallback if currently calibrating
    prompt_emotion = "Neutral" if emotion == "Calibrating..." else emotion
    
    print("[DEBUG] Pipeline Inputs:")
    print(f"  - Question: {question}")
    print(f"  - Retrieved Context Chunks Count: {len(docs)}")
    print(f"  - Injected Live Emotion: {emotion}")
    
    payload = {
        "context": format_docs(docs),
        "question": question,
        "student_emotion": prompt_emotion
    }
    
    # LCEL Pass-Through Trace
    print(f"[DEBUG] [LCEL Chain] Bound prompt payload state: {payload}")
    
    return payload

# The LCEL Pipeline
rag_chain = (
    RunnableParallel({
        "context_docs": retrieve_docs,
        "question": itemgetter("question"),
        "student_emotion": RunnableLambda(fetch_live_emotion)
    })
    | RunnableLambda(print_telemetry_and_format)
    | prompt
    | llm
    | StrOutputParser()
)

def parse_questions_response(raw_text: str) -> list[str]:
    """Robust extraction helper to parse JSON array of strings from LLM output"""
    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(item) for item in data[:3]]
    except Exception:
        pass
        
    lines = re.split(r'\n+', raw_text)
    questions = []
    for line in lines:
        line = line.strip()
        match = re.match(r'^(?:\d+[\.\)]|[\*\-\+])\s*(.+)$', line)
        if match:
            questions.append(match.group(1).strip())
        elif line and len(line) > 10 and line.endswith('?'):
            questions.append(line)
            
    questions = [q for q in questions if q]
    if len(questions) >= 3:
        return questions[:3]
        
    fallback_qs = [line.strip() for line in lines if line.strip() and len(line.strip()) > 5]
    if len(fallback_qs) >= 3:
        return fallback_qs[:3]
        
    return [
        "Explain the core concept of the algorithm discussed.",
        "Compare the performance of this method with alternatives.",
        "How would you implement this in practice?"
    ]

def aggregate_struggles(session_id: str) -> list[str]:
    """Compile a list of struggle points based on emotional spikes, rewinds, and chat questions"""
    spikes = db_logger.get_engagement_spikes(session_id)
    rewinds = db_logger.get_navigation_rewinds(session_id)
    chats = db_logger.get_chat_history(session_id)
    
    struggle_points = []
    
    for seg, count in spikes:
        desc = f"Student showed frustration or confusion {count} times during segment: '{seg}'"
        struggle_points.append(desc)
        db_logger.log_struggle(session_id, f"Emotional Spike ({count} times)", seg)
        
    for seg, count in rewinds:
        desc = f"Student scrubbed backward {count} times to segment: '{seg}'"
        struggle_points.append(desc)
        db_logger.log_struggle(session_id, f"Video Rewind ({count} times)", seg)
        
    for question, answer, emotion in chats:
        struggle_points.append(f"Student asked chat question: '{question}' while feeling {emotion}")
        
    return struggle_points

# Create the API Endpoints
@app.post("/log_engagement")
async def log_engagement(payload: EmotionLogRequest):
    print(f"[DIAGNOSTIC] RAG Service Ingest Point hit on route /log_engagement with payload: {payload.dict()}")
    """
    Receive passive emotional engagement data while the student watches
    the video, synchronize it with the transcript, and log it into the 
    database.
    """
    try:
        transcript_segment_dict = synchronizer.get_transcript_segment(
            payload.video_timestamp
        )

        transcript_text = transcript_segment_dict.get(
            "text", "[Silence/No dialogue]")

        db_logger.log_engagement(
            session_id=payload.session_id,
            video_timestamp=payload.video_timestamp,
            emotion_state=payload.emotion_state,
            transcript_segment=transcript_text
        )
        
        return {"status": "success", "message": "Engagement logged securely."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_navigation")
async def log_navigation(payload: NavigationLogRequest):
    print(f"[DIAGNOSTIC] RAG Service Ingest Point hit on route /log_navigation with payload: {payload.dict()}")
    """
    Receive video scrub/rewind events, synchronize the target timestamp, 
    and log the navigation event into the database.
    """
    try:
        segment_dict = synchronizer.get_transcript_segment(payload.timestamp_to)
        segment_text = segment_dict.get("text", "[Silence/No dialogue]")
        
        db_logger.log_navigation(
            session_id=payload.session_id,
            timestamp_from=payload.timestamp_from,
            timestamp_to=payload.timestamp_to,
            event_type=payload.event_type,
            transcript_segment=segment_text
        )
        return {"status": "success", "message": "Navigation event logged."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"[DIAGNOSTIC] RAG Service Ingest Point hit on route /chat with payload: {request.dict()}")
    print(f"\n--- DEBUG: Received Emotion: {request.student_emotion} ---\n")
    try:
        payload = {
            "question": request.question,
            "student_emotion": request.student_emotion
        }

        answer = rag_chain.invoke(payload)
        
        db_logger.log_chat(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            student_emotion=request.student_emotion or "neutral"
        )
        
        return {"answer": answer}
    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_qa")
async def generate_qa_endpoint(payload: QAGenerationRequest):
    """
    Compile struggles, retrieve relevant context, prompt the LLM, 
    and return 3 customized check questions.
    """
    try:
        struggle_points = aggregate_struggles(payload.session_id)
        struggles_count = len(struggle_points)
        
        if struggle_points:
            struggle_log_text = "\n".join(struggle_points)
        else:
            struggle_log_text = "No explicit struggles detected during the session."
            
        retrieved_docs = retriever.invoke(struggle_log_text)
        context_text = format_docs(retrieved_docs)
        
        qa_template = """System: You are an expert AI Research Assistant. Your task is to analyze the student's Struggle Log and generate 3 custom, high-quality, concept-check test questions. 
Target the specific topics or transcript segments where the student struggled (indicated by emotional spikes, video rewinds, or chat questions).
You must use ONLY the provided Context notes to ensure correct facts.
Format your output exactly as a JSON list of 3 strings, and nothing else.

Context Notes:
{context}

Struggle Log:
{struggle_log}

Questions (JSON array of 3 strings):"""
        
        qa_prompt = PromptTemplate.from_template(qa_template)
        formatted_prompt = qa_prompt.format(context=context_text, struggle_log=struggle_log_text)
        
        print(f"[DEBUG] [Aggregated Struggles Vector Count]: {struggles_count}")
        print(f"[DEBUG] [Retrieved Vector Space Chunks]: {len(retrieved_docs)}")
        print(f"[DEBUG] [LLM Payload Handoff]: {formatted_prompt}")
        
        raw_output = llm.invoke(formatted_prompt).content
        
        questions = parse_questions_response(raw_output)
        
        questions_json = json.dumps(questions)
        db_logger.save_post_video_questions(payload.session_id, questions_json)
        
        return {"questions": questions}
    except Exception as e:
        print(f"[ERROR] QA Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current_session_state")
def get_current_session_state(session_id: str = "default_session"):
    try:
        telemetry = db_logger.get_latest_telemetry(session_id)
        return telemetry
    except Exception as e:
        return {"playhead": 0.0, "emotion": "Neutral", "latency_ms": 0.0, "error": str(e)}

@app.get("/telemetry_health_check")
def telemetry_health_check(session_id: str = "default_session"):
    try:
        health = db_logger.get_telemetry_health(session_id)
        print(f"[DIAGNOSTIC] Health Check requested: {health['count']} active metrics stored.")
        return health
    except Exception as e:
        return {"count": 0, "latest_timestamp": 0.0, "error": str(e)}