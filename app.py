import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

import google.generativeai as genai
from google.generativeai import upload_file, get_file

import time
from pathlib import Path
import tempfile
import os

from dotenv import load_dotenv
load_dotenv()

# --- Configure Google API key ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Video AI Summarizer",
    page_icon="🎥",
    layout="wide"
)

# --- Custom CSS ---
CUSTOM_CSS = """
<style>
/* Main container width */
.block-container {
    max-width: 900px;
}

/* Title and header style */
h1, h2, h3 {
    font-family: "Helvetica Neue", sans-serif;
    margin-bottom: 0.3rem;
    font-weight: 700;
}

/* Modify text area to be a bit taller */
.stTextArea textarea {
    height: 6rem !important;
}

/* Button styling */
div.stButton > button {
    background-color: #FF4B4B;
    color: #FFFFFF;
    border-radius: 0.25rem;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 1.2rem;
    border: none;
    margin-top: 1rem;
}
div.stButton > button:hover {
    background-color: #DC3F3F;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #F8F9FB;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Sidebar: Basic Instructions ---
st.sidebar.title("Instructions")
st.sidebar.write(
    """
    1. **Upload** a video file (mp4, mov, avi).
    2. **Enter** a question or insight about the video.
    3. **Click** 'Analyze Video' to process.
    4. **View** your result below.
    """
)
st.sidebar.info("This app uses Gemini 2.0 Flash Exp and DuckDuckGo for enriched video analysis.")

# --- Main Title & Subtitle ---
st.title("AI Video Summarizer Agent")
st.subheader("Powered by Gemini 2.0 Flash Exp")

# --- Initialize the AI Agent ---
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

multimodal_Agent = initialize_agent()

st.divider()

# --- Video Uploader ---
video_file = st.file_uploader(
    "Upload your video here:",
    type=['mp4', 'mov', 'avi'],
    help="Choose a video file you wish to analyze."
)

# --- User Query ---
user_query = st.text_area(
    "Ask a question or describe what insights you need from the video:",
    placeholder="For example: 'Summarize the main points', or 'What is the speaker talking about?'",
)

analyze_button = st.button("🔍 Analyze Video")

# --- Main Process and Output ---
if video_file and analyze_button:
    if not user_query.strip():
        st.warning("Please enter a question or insight to analyze the video.")
    else:
        # Display the video immediately
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        
        st.video(video_path, format="video/mp4", start_time=0)

        try:
            with st.spinner("Processing video and gathering insights..."):
                # Upload and process the video with Google Generative AI
                processed_video = upload_file(video_path)
                while processed_video.state.name == "PROCESSING":
                    time.sleep(1)
                    processed_video = get_file(processed_video.name)

                # Prompt the AI agent
                analysis_prompt = f"""
                    Analyze the uploaded video for content and context.
                    Use additional web research if needed.
                    Question or Insight: {user_query}

                    Provide a detailed, user-friendly, and actionable response.
                """

                # AI agent response
                response = multimodal_Agent.run(
                    analysis_prompt,
                    videos=[processed_video]
                )

            st.divider()
            st.subheader("Analysis Result")
            st.markdown(response.content)

        except Exception as error:
            st.error(f"An error occurred during analysis: {error}")
        finally:
            Path(video_path).unlink(missing_ok=True)

elif not video_file:
    st.info("Please upload a video file to begin analysis.")
