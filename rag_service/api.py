from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import requests

from session_logger import SessionLogger
from synchronizer import TemporalSynchronizer

# 1. Initialize FastAPI
app = FastAPI(title="Lecture Analysis API")

TRANSCRIPT_PATH = os.path.join(os.path.dirname(__file__), "data",
                               "transcript.json")

try:
    synchronizer = TemporalSynchronizer(TRANSCRIPT_PATH)
    db_logger = SessionLogger()
    print("Synchronizer and Session Logger initialized successfully.")
except Exception as e:
    print(f"Initialization Error: {e}")

# Pydantic Models for Data Validation
# Define the expected data structure from the frontend
class EmotionLogRequest(BaseModel):
    video_timestamp: float
    emotion_state: str

class ChatRequest(BaseModel):
    question: str
    student_emotion: str | None = "neutral" # Defaults to neutral if the camera is off

# 3. Setup the LangChain Components globally so they load once
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOllama(model="gemma3:4b", temperature=0.2)

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
            return response.json().get("student_emotion", "Neutral")
    except Exception:
        pass
    return inputs.get("student_emotion", "Neutral") or "Neutral"

def retrieve_docs(inputs):
    return retriever.invoke(inputs["question"])

def print_telemetry_and_format(inputs):
    question = inputs["question"]
    docs = inputs["context_docs"]
    emotion = inputs["student_emotion"]
    
    print("[DEBUG] Pipeline Inputs:")
    print(f"  - Question: {question}")
    print(f"  - Retrieved Context Chunks Count: {len(docs)}")
    print(f"  - Injected Live Emotion: {emotion}")
    
    return {
        "context": format_docs(docs),
        "question": question,
        "student_emotion": emotion
    }

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

# Create the API Endpoints
@app.post("/log_engagement")
async def log_engagement(payload: EmotionLogRequest):
    """
    Receive passive emotional engagement data while the student watches
    the video, synchronize it with the transcript, and log it into the 
    database.
    """
    try:
        # Find what was being said at the exact moment
        transcript_segment_dict = synchronizer.get_transcript_segment(
            payload.video_timestamp
        )

        transcript_text = transcript_segment_dict.get(
            "text", "[Silence/No dialogue]")

        # Write the synchronized data to SQLite
        db_logger.log_engagement(
            video_timestamp=payload.video_timestamp,
            emotion_state=payload.emotion_state,
            transcript_segment=transcript_text
        )
        
        return {"status": "success", "message": "Engagement logged securely."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"\n--- DEBUG: Received Emotion: {request.student_emotion} ---\n")
    try:
        # Package the data into a dict for the LCEL chain
        payload = {
            "question": request.question,
            "student_emotion": request.student_emotion
        }

        # Run the pipeline with the full payload
        answer = rag_chain.invoke(payload)
        return {"answer": answer}
    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))