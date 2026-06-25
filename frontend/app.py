import streamlit as st
import requests
import json
import os
import time

# Set up the UI page
st.set_page_config(page_title="Lecture Analysis RAG Tutor", page_icon="🎙️")
st.title("Lecture Analysis RAG Tutor")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "post_video_questions" not in st.session_state:
    st.session_state.post_video_questions = []

# Sidebar configurations and telemetry
st.sidebar.header("Session Settings")
st.session_state.session_id = st.sidebar.text_input(
    "Active Session ID", value=st.session_state.session_id
)

# Render hidden input to receive telemetry from Javascript bridge
telemetry_json = st.text_input(
    "Telemetry Bridge Input",
    value="",
    label_visibility="collapsed",
    key="telemetry_bridge"
)

# Parse bridge telemetry
playhead_pos = 0.0
latency_ms = 0.0
extracted_emotion = "Neutral"

if telemetry_json:
    try:
        telemetry_data = json.loads(telemetry_json)
        playhead_pos = telemetry_data.get("playhead", 0.0)
        latency_ms = telemetry_data.get("latency_ms", 0.0)
        extracted_emotion = telemetry_data.get("emotion", "Neutral")
    except Exception:
        pass

# Calibration Status
calibration_progress = 0.0
try:
    state_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "emotion_service", "current_state.json"))
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state_data = json.load(f)
            calibration_progress = state_data.get("calibration_progress", 0.0)
except Exception:
    pass

st.sidebar.markdown("### Calibration Status")
if calibration_progress < 1.0:
    st.sidebar.progress(calibration_progress)
    st.sidebar.write(f"Calibrating: {int(calibration_progress * 100)}%")
else:
    st.sidebar.info("Calibration Complete")

# Rolling Telemetry Dashboard
st.sidebar.markdown("### Researcher Telemetry Dashboard")
st.sidebar.markdown(f"**Active Session ID:** `{st.session_state.session_id}`")
st.sidebar.markdown(f"**Current Playhead Position:** `{playhead_pos:.1f}s`")
st.sidebar.markdown(f"**Extracted Emotion State:** `{extracted_emotion}`")
st.sidebar.markdown(f"**Queue Latency Metric:** `{latency_ms:.1f}ms`")

# Custom HTML5 Video Player
video_html = """
<video id="lecture-video" width="100%" height="auto" controls>
    <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
<script>
    const video = document.getElementById('lecture-video');
    let previousTime = 0;
    let lastTimeUpdate = 0;
    let seekTimeout = null;

    video.addEventListener('timeupdate', () => {
        if (!video.seeking) {
            previousTime = video.currentTime;
        }
        
        const now = Date.now();
        if (now - lastTimeUpdate >= 1000) {
            lastTimeUpdate = now;
            window.parent.postMessage({
                type: 'video_telemetry',
                event: 'engagement',
                timestamp: now,
                payload: {
                    video_timestamp: video.currentTime
                }
            }, '*');
        }
    });

    video.addEventListener('seeked', () => {
        if (seekTimeout) clearTimeout(seekTimeout);
        const now = Date.now();
        seekTimeout = setTimeout(() => {
            const toTime = video.currentTime;
            const fromTime = previousTime;
            const type = toTime < fromTime ? 'rewind' : 'forward';
            
            window.parent.postMessage({
                type: 'video_telemetry',
                event: 'navigation',
                timestamp: now,
                payload: {
                    timestamp_from: fromTime,
                    timestamp_to: toTime,
                    event_type: type
                }
            }, '*');
            previousTime = toTime;
        }, 500);
    });
</script>
"""

# Web Messaging Bridge Script in Parent
js_bridge = f"""
<script>
window.addEventListener('message', (event) => {{
    if (event.data && event.data.type === 'video_telemetry') {{
        const payload = event.data.payload;
        const sessionId = '{st.session_state.session_id}';
        const eventTime = event.data.timestamp;
        const latency = Date.now() - eventTime;
        
        if (event.data.event === 'engagement') {{
            fetch('http://localhost:8001/current_emotion')
                .then(res => res.json())
                .then(emotionData => {{
                    const emotion = emotionData.student_emotion || 'Neutral';
                    
                    fetch('http://localhost:8000/log_engagement', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            session_id: sessionId,
                            video_timestamp: payload.video_timestamp,
                            emotion_state: emotion
                        }})
                    }});
                    
                    const input = window.parent.document.querySelector('input[aria-label="Telemetry Bridge Input"]');
                    if (input) {{
                        input.value = JSON.stringify({{
                            playhead: payload.video_timestamp,
                            latency_ms: latency,
                            emotion: emotion
                        }});
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    }}
                }}).catch(() => {{
                    fetch('http://localhost:8000/log_engagement', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            session_id: sessionId,
                            video_timestamp: payload.video_timestamp,
                            emotion_state: 'Neutral'
                        }})
                    }});
                    
                    const input = window.parent.document.querySelector('input[aria-label="Telemetry Bridge Input"]');
                    if (input) {{
                        input.value = JSON.stringify({{
                            playhead: payload.video_timestamp,
                            latency_ms: latency,
                            emotion: 'Neutral'
                        }});
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    }}
                }});
        }} else if (event.data.event === 'navigation') {{
            fetch('http://localhost:8000/log_navigation', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    session_id: sessionId,
                    timestamp_from: payload.timestamp_from,
                    timestamp_to: payload.timestamp_to,
                    event_type: payload.event_type
                }})
            }});
            
            const input = window.parent.document.querySelector('input[aria-label="Telemetry Bridge Input"]');
            if (input) {{
                input.value = JSON.stringify({{
                    playhead: payload.timestamp_to,
                    latency_ms: latency,
                    emotion: 'Seeking'
                }});
                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
        }}
    }}
}});
</script>
"""

# Render Bridge JS
st.markdown(js_bridge, unsafe_allow_html=True)

# Main UI Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Lecture Media Player")
    st.components.v1.html(video_html, height=360)

with col2:
    st.subheader("Pedagogical Chat Assistant")
    chat_container = st.container(height=350)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# React to user chat input
if prompt := st.chat_input("Ask a question about the lecture..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Query current emotion from the emotion state API
    try:
        emotion_res = requests.get("http://localhost:8001/current_emotion", timeout=1.0)
        curr_emotion = emotion_res.json().get("student_emotion", "Neutral")
    except Exception:
        curr_emotion = "Neutral"
        
    # Send request to core RAG service
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "question": prompt,
            "student_emotion": curr_emotion
        }
        res = requests.post("http://localhost:8000/chat", json=payload, timeout=10.0)
        res.raise_for_status()
        answer = res.json().get("answer", "No response.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        
    st.rerun()

# Post-Video Q&A Block
st.markdown("---")
st.subheader("Lecture Assessment")
if st.button("Complete Session & Generate Q&A"):
    with st.spinner("Analyzing student struggles and synthesizing custom questions..."):
        try:
            payload = {
                "session_id": st.session_state.session_id
            }
            res = requests.post("http://localhost:8000/generate_qa", json=payload, timeout=15.0)
            res.raise_for_status()
            questions = res.json().get("questions", [])
            st.session_state.post_video_questions = questions
        except Exception as e:
            st.error(f"Failed to generate Q&A: {e}")

if st.session_state.post_video_questions:
    st.info("Based on your struggle telemetry, answer the following concept check questions:")
    for idx, question in enumerate(st.session_state.post_video_questions):
        st.markdown(f"**Question {idx + 1}:** {question}")