import streamlit as st
import requests
import json
import os
import time

# Set up the UI page
st.set_page_config(page_title="Lecture Analysis RAG Tutor", page_icon="🎙️")
st.title("Lecture Analysis RAG Tutor")

# Inject custom CSS to disable the fade-out/flicker effect when st.fragment reruns
st.markdown(
    """
    <style>
    /* Prevent fading/opacity changes on fragment reruns and stale states */
    [data-testid="stFragment"],
    [data-testid="stFragment"] *,
    [data-st-mode="stale"],
    [data-st-mode="stale"] *,
    [data-st-mode="running"],
    [data-st-mode="running"] * {
        opacity: 1 !important;
        transition: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "post_video_questions" not in st.session_state:
    st.session_state.post_video_questions = []

# Initialize telemetry cache in session state so that it doesn't
# on every redraw
if "last_playhead" not in st.session_state:
    st.session_state.last_playhead = 0.0
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = "Neutral"
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0
if "last_telemetry_poll" not in st.session_state:
    st.session_state.last_telemetry_poll = 0.0
if "calibration_progress" not in st.session_state:
    st.session_state.calibration_progress = 0.0

if "last_calibration_progress" not in st.session_state:
    st.session_state.last_calibration_progress = 0.0
if "last_calibration_change_time" not in st.session_state:
    st.session_state.last_calibration_change_time = time.time()
if "calibration_is_stale" not in st.session_state:
    st.session_state.calibration_is_stale = False
if "live_telemetry_received" not in st.session_state:
    st.session_state.live_telemetry_received = False
if "last_api_change_time" not in st.session_state:
    st.session_state.last_api_change_time = time.time()
if "last_raw_scores" not in st.session_state:
    st.session_state.last_raw_scores = {}
if "api_is_stale" not in st.session_state:
    st.session_state.api_is_stale = False

# Sidebar configurations and telemetry
st.sidebar.header("Session Settings")
st.session_state.session_id = st.sidebar.text_input(
    "Active Session ID", value=st.session_state.session_id
)

@st.fragment(run_every=1.0)
def show_sidebar_dashboard():
    # Fetch latest calibration and telemetry data from services
    now_poll = time.time()
    st.session_state.last_telemetry_poll = now_poll
    
    # 1. Fetch emotion and calibration data
    live_telemetry_success = False
    api_emotion = "Neutral"
    api_calibration_progress = 0.0
    api_raw_scores = {}
    
    try:
        emotion_res = requests.get("http://localhost:8001/current_emotion", timeout=1.0)
        if emotion_res.status_code == 200:
            emotion_data = emotion_res.json()
            api_emotion = emotion_data.get("student_emotion", "Neutral")
            api_calibration_progress = emotion_data.get("calibration_progress", 0.0)
            api_raw_scores = emotion_data.get("raw_scores", {})
            live_telemetry_success = True
    except Exception as e:
        print(f"[DIAGNOSTIC] Frontend Polling Failure on port 8001: {e}")
        
    # Check if calibration progress is stale and if api telemetry is stale
    if live_telemetry_success:
        # Detect active changes in calibration or raw scores
        has_changed = False
        if api_calibration_progress != st.session_state.last_calibration_progress:
            has_changed = True
            st.session_state.last_calibration_progress = api_calibration_progress
            st.session_state.last_calibration_change_time = time.time()
            st.session_state.calibration_is_stale = False
        else:
            # If calibration is less than 1.0 and has not changed, check for staleness
            if api_calibration_progress < 1.0:
                elapsed_since_change = time.time() - st.session_state.last_calibration_change_time
                if elapsed_since_change > 5.0:
                    st.session_state.calibration_is_stale = True
            else:
                st.session_state.calibration_is_stale = False
                
        if api_raw_scores != st.session_state.last_raw_scores:
            has_changed = True
            st.session_state.last_raw_scores = api_raw_scores
            
        if has_changed:
            st.session_state.last_api_change_time = time.time()
            st.session_state.api_is_stale = False
        else:
            if time.time() - st.session_state.last_api_change_time > 5.0:
                st.session_state.api_is_stale = True
                
        # We consider telemetry "live" if the API is active, not stale, and calibration is not stale
        if (not st.session_state.api_is_stale) and (not st.session_state.calibration_is_stale):
            st.session_state.live_telemetry_received = True
            st.session_state.calibration_progress = api_calibration_progress
            st.session_state.last_emotion = api_emotion
        else:
            st.session_state.live_telemetry_received = False
    else:
        st.session_state.live_telemetry_received = False
        
    # 2. Fetch session state / playhead telemetry from RAG service
    try:
        state_res = requests.get(
            f"http://localhost:8000/current_session_state?session_id={st.session_state.session_id}", 
            timeout=1.0
        )
        if state_res.status_code == 200:
            state_data = state_res.json()
            new_playhead = float(state_data.get("playhead", st.session_state.last_playhead))
            new_latency = float(state_data.get("latency_ms", st.session_state.last_latency))
            st.session_state.last_playhead = new_playhead
            st.session_state.last_latency = new_latency
            
            if not st.session_state.live_telemetry_received:
                new_emotion = state_data.get("emotion", st.session_state.last_emotion)
                st.session_state.last_emotion = new_emotion
    except Exception as e:
        print(f"[DIAGNOSTIC] Frontend Polling Failure on port 8000: {e}")

    # Check for stale state / offline live camera stream fallback
    current_state_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "emotion_service", "current_state.json"))
    file_fallback = False

    if os.path.exists(current_state_path):
        mtime = os.path.getmtime(current_state_path)
        if time.time() - mtime > 5.0:
            file_fallback = True
    else:
        file_fallback = True

    # Fallback decision
    is_fallback = file_fallback
    if st.session_state.live_telemetry_received:
        is_fallback = False
    if st.session_state.calibration_is_stale:
        is_fallback = True

    # Map variables to current session state values
    playhead_pos = st.session_state.last_playhead
    extracted_emotion = st.session_state.last_emotion
    latency_ms = st.session_state.last_latency

    if is_fallback:
        extracted_emotion = "Live Feed Offline (Defaulting to Neutral)"
        calibration_progress_val = 1.0
    else:
        calibration_progress_val = st.session_state.calibration_progress

    st.markdown("### Calibration Status")
    if is_fallback:
        st.progress(1.0)
        st.write("Simulation Mode Active")
    else:
        if calibration_progress_val < 1.0:
            st.progress(calibration_progress_val)
            st.write(f"Calibrating: {int(calibration_progress_val * 100)}%")
        else:
            st.info("Calibration Complete")

    # Rolling Telemetry Dashboard
    st.markdown("### Researcher Telemetry Dashboard")
    st.markdown(f"**Active Session ID:** `{st.session_state.session_id}`")
    st.markdown(f"**Current Playhead Position:** `{playhead_pos:.1f}s`")
    st.markdown(f"**Extracted Emotion State:** `{extracted_emotion}`")
    st.markdown(f"**Queue Latency Metric:** `{latency_ms:.1f}ms`")

# Render the sidebar dashboard fragment inside the sidebar context
with st.sidebar:
    show_sidebar_dashboard()


video_html = f"""
<video id="lecture-video" width="100%" height="auto" controls>
    <source src="http://localhost:8000/video" type="video/mp4">
    Your browser does not support the video tag.
</video>
<script>
    const video = document.getElementById('lecture-video');
    let previousTime = 0;
    let lastTimeUpdate = 0;
    let seekTimeout = null;
    const sessionId = '{st.session_state.session_id}';

    video.addEventListener('timeupdate', () => {{
        if (!video.seeking) {{
            previousTime = video.currentTime;
        }}
        
        const now = Date.now();
        if (now - lastTimeUpdate >= 1000) {{
            lastTimeUpdate = now;
            
            fetch('http://localhost:8001/current_emotion')
                .then(res => res.json())
                .then(emotionData => {{
                    const emotion = emotionData.student_emotion || 'Neutral';
                    fetch('http://localhost:8000/log_engagement', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            session_id: sessionId,
                            video_timestamp: video.currentTime,
                            emotion_state: emotion
                        }})
                    }});
                }}).catch(() => {{
                    fetch('http://localhost:8000/log_engagement', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            session_id: sessionId,
                            video_timestamp: video.currentTime,
                            emotion_state: 'Neutral'
                        }})
                    }});
                }});
        }}
    }});

    video.addEventListener('seeked', () => {{
        if (seekTimeout) clearTimeout(seekTimeout);
        const now = Date.now();
        seekTimeout = setTimeout(() => {{
            const toTime = video.currentTime;
            const fromTime = previousTime;
            const type = toTime < fromTime ? 'rewind' : 'forward';
            
            fetch('http://localhost:8000/log_navigation', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    session_id: sessionId,
                    timestamp_from: fromTime,
                    timestamp_to: toTime,
                    event_type: type
                }})
            }});
            previousTime = toTime;
        }}, 500);
    }});
</script>
"""

# Main UI Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Lecture Media Player")
    st.iframe(video_html, height=360)

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
        res = requests.post("http://localhost:8000/chat", json=payload, timeout=60.0)
        res.raise_for_status()
        answer = res.json().get("answer", "No response.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except requests.exceptions.Timeout:
        st.session_state.messages.append({"role": "assistant", "content": "[SYSTEM TELEMETRY] Inference engine busy or cold-starting. Retrying connection context..."})
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
            res = requests.post("http://localhost:8000/generate_qa", json=payload, timeout=90.0)
            res.raise_for_status()
            questions = res.json().get("questions", [])
            st.session_state.post_video_questions = questions
        except requests.exceptions.Timeout:
            st.error("[SYSTEM TELEMETRY] Inference engine busy or cold-starting. Retrying connection context...")
        except Exception as e:
            st.error(f"Failed to generate Q&A: {e}")

if st.session_state.post_video_questions:
    st.info("Based on your struggle telemetry, answer the following concept check questions:")
    for idx, question in enumerate(st.session_state.post_video_questions):
        st.markdown(f"**Question {idx + 1}:** {question}")

# Periodic updates are handled automatically by the show_sidebar_dashboard fragment