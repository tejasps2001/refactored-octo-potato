# Emotionally Aware RAG Tutor
## Status: Work in Progress

A localized RAG pipeline that integrates computer vision to understand student engagement and provide tailored educational responses. 
## Overview

This system combines a local Retrieval-Augmented Generation (RAG) backend with a computer vision service that tracks facial blendshapes. By analyzing emotional states like frustration or concentration, the tutor adapts its pedagogical tone to better support student learning. 
## Features

* **Local RAG Pipeline**: Uses LangChain Expression Language (LCEL) and ChromaDB for privacy-focused document querying. 

* **Emotion Recognition**: MediaPipe-based service that calculates engagement metrics in real-time. 

* **Decoupled Architecture**: Distributed microservice design featuring a FastAPI backend and a Streamlit frontend. 

* **Cross-Service Communication**: Future-ready API contracts that allow the vision service to inject emotional context into the LLM prompt. 

## Tech Stack
| Category |	Technology |
| -------- | ----------- |
| Language | Python 3.13 |
| LLM / Embeddings	| Ollama (Gemma 3 4B), Nomic Embed |
| Orchestration	| LangChain (LCEL) |
| API Framework	| FastAPI |
| Frontend	| Streamlit |
| Vision/ML	| MediaPipe, TensorFlow |
| Database|	ChromaDB |


## Getting Started
### Prerequisites
* Linux (Developed on) or Windows (Compatible). 
* Python 3.10+ installed.
* Ollama with `gemma3:4b` and `nomic-embed-text` models. 

### Installation

1. **Clone the repository**: `git clone https://github.com/tejasps2001/refactored-octo-potato/tree/master`

2. **Initialize Environments**: Use the provided bootstrap scripts to create isolated virtual environments for both the RAG and Emotion services.
    * `./emotion_service/run.sh `
    * `./run_rag.sh`

## Roadmap

  *  [x] Initial project setup and RAG script prototype. 

   * [x] Microservice refactoring (FastAPI + Streamlit). 

   * [x] Automated environment bootstrapping and process management. 

   * [ ] Integration of Emotion Service with RAG API (In Progress). 

   * [ ] Fine-tuning of local LLM for educational pedagogy. 

## Contributing

Contributions are welcome. Please ensure that all dependencies are updated in the relevant ```requirements.txt``` files and avoid committing system-specific binaries or databases by respecting the ```.gitignore```. 
## License

This project is licensed under the MIT License.
