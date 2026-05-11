import streamlit as st
import requests

# Set up the UI page
st.set_page_config(page_title="Local RAG Chat", page_icon="🤖")
st.title("Local Document Assistant")
st.caption("Powered by Gemma3:4b and FastAPI")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on UI refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask something about your documents..."):
    
    # 1. Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display an empty assistant message while waiting
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Checking student engagement & thinking...")
        
        # Fetch emotion state
        try:
            # Ping the tiny emotion API (Timeout set to 2 seconds so it doesn't hang)
            emotion_res = requests.get("http://localhost:8001/current_emotion", timeout=2)
            current_emotion = emotion_res.json().get("student_emotion", "neutral")
        except requests.exceptions.RequestException:
            # If the emotion API is off, just default to neutral and keep going
            current_emotion = "neutral"
        try:
            payload = {
                "question": prompt,
                "student_emotion": current_emotion
            }
            # Send the request to your FastAPI backend
            response = requests.post(
                "http://localhost:8000/chat", 
                json=payload
            )
            response.raise_for_status() # Check for HTTP errors
            
            # Extract the answer
            answer = response.json().get("answer", "No answer received.")
            
            # Update the UI
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except requests.exceptions.ConnectionError:
            message_placeholder.error("Error: Could not connect to the backend. Is api.py running?")
        except Exception as e:
            message_placeholder.error(f"An error occurred: {e}")