import streamlit as st
import os
import whisper
import cohere
import shutil
import textwrap
from datetime import datetime
from dotenv import load_dotenv
from fpdf import FPDF

# ========== Config ==========
load_dotenv()
CO_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(CO_API_KEY)

st.set_page_config(page_title="YouTube Notes Generator", layout="centered")
st.title("📝 YouTube Notes Generator")

# ========== Constants ==========
MODEL_SIZE = "tiny"
RUNS_DIR = "runs"
CHUNK_WORD_LIMIT = 3000

# ========== Inputs ==========
st.markdown("Upload an audio file (`.mp3`, `.wav`, or `.webm`) for transcription and summarization.")
audio_file = st.file_uploader("📤 Upload Audio File", type=["mp3", "wav", "webm"])

@st.cache_resource
def load_model():
    return whisper.load_model(MODEL_SIZE)
model = load_model()

# ========== Helpers ==========

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

# ========== Main Action ==========
if st.button("📝 Generate Notes"):
    if not audio_file:
        st.error("Please upload an audio file.")
    else:
        run_dir = os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)

        try:
            audio_path = os.path.join(run_dir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())

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
            st.error("Something went wrong.")
            st.exception(e)

        finally:
            try:
                shutil.rmtree(run_dir)
                st.info("Temporary files cleaned.")
            except:
                pass
