from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

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

template = """You are a helpful teaching assistant. Use the retrieved context to answer the student's question. 
CRITICAL INSTRUCTION: The camera detects that the student is currently feeling: {student_emotion}.
If they are frustrated, be extra patient, break down the steps clearly, and offer encouragement.
If they are engaged, provide a concise, technical answer.

Context:
{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The LCEL Pipeline
rag_chain = (
    {
        # Grab the question from the RunnableParallel, pass it to the retriever, then format the docs
        "context": itemgetter("question") | retriever | format_docs,
        # Grab the question and pass it straight through
        "question": itemgetter("question"),
        # Grab the emotion and pass it straight through
        "student_emotion": itemgetter("student_emotion")
    }
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