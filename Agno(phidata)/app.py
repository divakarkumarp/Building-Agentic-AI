import streamlit as st
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.google import Gemini
from google.generativeai import upload_file,get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(page_title=" AI Agent", page_icon="📹", layout="wide")

st.title("Phidata Video Summarizer AI Agent 🎥")
st.header("Powered by Gemini 2.0 Flash Exp")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        role="You are a video summarizer. You will be given a video and you will summarize it.",
        #goals=["Summarize the video in 3-5 sentences"],
        #tools=[DuckDuckGo(), Gemini()],
        model=Gemini(id="gemini-2.0-flash-exp"),
        markdown=True,
    )

# Initialize the agent
multimodal_agent = initialize_agent()

# File uploader
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], help="Upload a video file to summarize")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.video(video_path, format="video/mp4", start_time=0)
    
    user_query = st.text_input("What insights are you seeking from the video?", 
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
    )
    if st.button("Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    processed_video = upload_file(video_path)
                    with processed_video.state.name == "PROCESSING":
                        st.write("Video is being processed. Please wait...")
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                    
                    # Prompt generation for Analysis
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    # AI Agent Processing
                    response = multimodal_agent.run(analysis_prompt, video=[processed_video])

                # Display the result
                st.subheader("AI Agent Response:")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred: {error}")
                st.error("Please try again with a different video or query.")
            
            finally:
                # CLean up the temporary video file
                Path(video_path).unlink(missing_ok=True)

    else:
        st.info("Upload a video file to begin analysis.")    
    
    # CUstomize test area height
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



                
                