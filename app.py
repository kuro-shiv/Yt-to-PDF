import streamlit as st
import os
import yt_dlp
import whisper
import cohere
import shutil
import textwrap
from datetime import datetime
from dotenv import load_dotenv
from fpdf import FPDF

# ========== Config ========== #
load_dotenv()
CO_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(CO_API_KEY)

st.set_page_config(page_title="YouTube Notes Generator", layout="centered")
st.title("📝 YouTube Notes Generator")

# ========== Constants ========== #
MODEL_SIZE = "tiny"
RUNS_DIR = "runs"
CHUNK_WORD_LIMIT = 3000

@st.cache_resource
def load_model():
    return whisper.load_model(MODEL_SIZE)

model = load_model()

# ========== Input ========== #
video_url = st.text_input("📎 Enter YouTube video URL:")
st.text("Max video length is 30 mins")

# ========== Helper Functions ========== #
def download_audio(url, output_dir):
    output_path = os.path.join(output_dir, 'audio.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'noplaylist': True,
        'user_agent': 'Mozilla/5.0',
        'http_headers': {
            'Accept-Language': 'en-US,en;q=0.9'
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for file in os.listdir(output_dir):
        if file.startswith("audio."):
            return os.path.join(output_dir, file)

def split_text(text, max_words=CHUNK_WORD_LIMIT):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield ' '.join(words[i:i + max_words])

def summarize_with_cohere(text):
    summaries = []
    for idx, chunk in enumerate(split_text(text)):
        with st.spinner(f"✍️ Summarizing chunk {idx + 1}..."):
            response = co.summarize(
                text=chunk,
                format="bullets",
                length="long",
                extractiveness="medium",
                temperature=0.3
            )
            summaries.append(response.summary)
    return "\n\n".join(summaries)

def generate_pdf(text, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in textwrap.wrap(text, width=100):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

# ========== Main ========== #
if st.button("📝 Summarize in Notes"):
    if not video_url:
        st.error("Please enter a YouTube URL.")
        
    else:
        run_dir = os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)

        try:
            audio_path = download_audio(video_url, run_dir)

            with st.spinner("🎤 Transcribing audio..."):
                transcript = model.transcribe(audio_path)["text"]

            notes = summarize_with_cohere(transcript)

            st.success("Notes generated!")
            st.subheader("🗒️ Structured Notes")
            st.markdown(notes)

            pdf_path = os.path.join(run_dir, "notes.pdf")
            generate_pdf(notes, pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download Notes as PDF", f, file_name="notes.pdf", mime="application/pdf")
            st.success("✅ PDF ready!")

        except Exception as e:
            st.error("Something went wrong. Please try again.")
            st.exception(e)

        finally:
            try:
                shutil.rmtree(run_dir)
                st.info("Temporary files cleaned.")
            except:
                pass

    st.markdown("---")

st.markdown("Contact: smartfresherhubsa@gmail.com | Phone: +91 8299142475 © 2025 Smart Fresher Hub")
