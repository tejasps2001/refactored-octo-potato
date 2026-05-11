from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize FastAPI
app = FastAPI(title="RAG API Backend")

# 2. Define the expected data structure from the frontend
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

# 4. Create the API Endpoint
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