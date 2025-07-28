import streamlit as st
import os
from datetime import datetime
import re
import yt_dlp
import whisper
from dotenv import load_dotenv
from fpdf import FPDF
import cohere
import shutil
import textwrap

# --- Load environment variables ---
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# --- Streamlit Config ---
st.set_page_config(page_title="YouTube Notes Generator", layout="centered")
st.title("📝 YouTube Notes Generator")

video_url = st.text_input("📎 Enter YouTube video URL:")

# --- Load Whisper Model (Tiny) ---
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")
model = load_model()

# --- Create Unique Folder ---
run_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)

# --- Download YouTube Audio ---
def download_audio(url, run_dir):
    output_path = os.path.join(run_dir, 'audio.webm')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

# --- Extract Video ID ---
def get_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# --- Split transcript into chunks ---
def split_text(text, max_words=3000):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield ' '.join(words[i:i + max_words])

# --- Cohere Note-style Summarization ---
def summarize_with_cohere_chunks(transcript_text):
    all_chunks = list(split_text(transcript_text))
    all_notes = []

    for idx, chunk in enumerate(all_chunks):
        try:
            with st.spinner(f"✍️ Summarizing chunk {idx + 1}/{len(all_chunks)}..."):
                response = co.summarize(
                    text=chunk,
                    format="bullets",
                    length="long",
                    extractiveness="medium",
                    temperature=0.3
                )
                all_notes.append(response.summary)
        except Exception as e:
            all_notes.append(f"[ERROR in chunk {idx + 1}]: {e}")
    return "\n\n".join(all_notes)

# --- Generate PDF from Notes ---
def generate_pdf(text, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in textwrap.wrap(text, width=100):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)
    return output_path

# --- Main Processing ---
if st.button("📝 Summarize in Notes"):
    if not video_url:
        st.error("❗ Please enter a valid YouTube video URL.")
    else:
        video_id = get_video_id(video_url)
        if not video_id:
            st.error("❌ Could not extract video ID from the URL.")
        else:
            try:
                st.info("📥 Downloading audio from YouTube...")
                audio_path = download_audio(video_url, run_dir)

                st.info("🎤 Transcribing with Whisper Tiny — may take ~1–2 mins")
                with st.spinner("⏳ Transcribing audio..."):
                    result = model.transcribe(audio_path)
                    transcript = result["text"]

                with open(os.path.join(run_dir, "transcript.txt"), "w", encoding="utf-8") as f:
                    f.write(transcript)

                st.info("🧠 Generating structured notes using Cohere...")
                notes = summarize_with_cohere_chunks(transcript)

                if notes.startswith("ERROR"):
                    st.error(notes)
                else:
                    st.success("📚 Notes generated successfully!")
                    st.subheader("🗒️ Structured Notes")
                    st.markdown(notes)

                    pdf_path = os.path.join(run_dir, "summary_notes.pdf")
                    generate_pdf(notes, pdf_path)

                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Download Notes as PDF",
                            data=f,
                            file_name="summary_notes.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
                    st.success("✅ Notes ready for download!")

            except Exception as e:
                st.error(f"❌ Error: {e}")

            # --- Clean Up Temporary Folder ---
            try:
                shutil.rmtree(run_dir)
                st.info("🧹 Cleaned up temporary files.")
            except Exception as e:
                st.warning(f"⚠️ Failed to delete temp folder: {e}")
