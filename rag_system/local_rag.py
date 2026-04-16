import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# Phase 1: Ingestion & Chunking
# ---------------------------------------------------------
# 1. Load your document (replace with the path to a real PDF on your computer)
# For testing, create a dummy text file if you don't have a PDF and use TextLoader instead.
loader = PyPDFLoader("MCA Internship 2026 Abstract-1.pdf")
docs = loader.load()

# 2. Split the document into smaller, digestible chunks for the LLM
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Number of characters per chunk
    chunk_overlap=50,     # Overlap to preserve context between chunks
)
splits = text_splitter.split_documents(docs)
print(f"Number of splits: {len(splits)}")
# Check if splits is empty
if len(splits) == 0:
    print("Error: No text was extracted from the documents.")
    # TODO: Use UnstructredLoader if it is a scanned PDF. If not use the PyPDFLoader. Check the type of PDF first
    sys.exit("Exiting: Please provide a text-based PDF or use an OCR loader for scanned PDFs.")

# ---------------------------------------------------------
# Phase 2: The Vector Store (ChromaDB)
# ---------------------------------------------------------
# 3. Initialize the local embedding model you downloaded via Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Create the Vector Database and embed the chunks
# Chroma will run entirely in your local memory for this script.
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 5. Create a Retriever from the database
# "similarity" search will find the chunks most mathematically similar to the user's question
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ---------------------------------------------------------
# Phase 3: The LLM and Prompt
# ---------------------------------------------------------
# 6. Initialize your Gemma model via Ollama
llm = ChatOllama(model="gemma3:4b", temperature=0.2)

# 7. Create a custom Prompt Template
# This dictates exactly how the LLM should behave using the retrieved data.
template = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

Context:
{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

# Helper function to format the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------------------------------------
# Phase 4: The LCEL Pipeline
# ---------------------------------------------------------
# 8. Build the pipeline using LangChain Expression Language (LCEL)
# The "|" symbol passes the output of the left side as the input to the right side.
rag_chain = (
    # Step A: Get the context from the retriever, and pass the question straight through
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # Step B: Feed both into the Prompt Template
    | prompt
    # Step C: Send the formatted prompt to the LLM
    | llm
    # Step D: Parse the messy LLM output into a clean string
    | StrOutputParser()
)

# ---------------------------------------------------------
# Phase 5: Execution
# ---------------------------------------------------------
# 9. Ask your question!
question = "What is the main topic of the document?"
print(f"Question: {question}\n")
print("Thinking...\n")

# Invoke the chain
response = rag_chain.invoke(question)
print(f"Answer: {response}")
