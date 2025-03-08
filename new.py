import streamlit as st
import openai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
from gtts import gTTS
import os
import tempfile
import dotenv


# Load environment variables from a .env file
dotenv.load_dotenv()

openai.api_key = st.secrets["OPENAI_API_KEY"]


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc])

# Chunking function
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

# Generate embeddings
def embed_text(texts):
    return np.array(embedding_model.encode(texts))

# Create FAISS index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Retrieve relevant chunks
def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    query_embedding = embed_text([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Generate AI Response
def generate_response(query, context):
    prompt = f"Answer the question based on the following context:\n\n{' '.join(context)}\n\nQuestion: {query}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

# Speech-to-Text
def speech_to_text(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
    
    model = WhisperModel("base", device="cpu")
    segments, _ = model.transcribe(temp_audio_path)
    transcription = " ".join(segment.text for segment in segments)
    os.remove(temp_audio_path)
    return transcription if transcription.strip() else "No speech detected."

# Text-to-Speech
def text_to_speech(text):
    tts = gTTS(text)
    tts_file = "response.mp3"
    tts.save(tts_file)
    return tts_file

# Callback function to handle query and reset input
def handle_query():
    if "text_query" in st.session_state and st.session_state.text_query:
        user_query = st.session_state.text_query
        
        # Retrieve relevant chunks and generate response
        relevant_chunks = retrieve_relevant_chunks(user_query, st.session_state.faiss_index, st.session_state.chunks)
        response_text = generate_response(user_query, relevant_chunks)

        # Convert response to voice if needed
        speech_file_path = text_to_speech(response_text)

        # Save to conversation history
        st.session_state.conversation_history.append({
            "user_query": user_query,
            "ai_response": response_text,
            "ai_response_voice": speech_file_path
        })
        
        # Clear the text input
        st.session_state.text_query = ""

def main():
    st.set_page_config(page_title="PDF RAG Chat", layout="wide")
    st.title("PDF RAG Chat with Voice & Text")

    # Sidebar: Upload PDF
    with st.sidebar:
        st.header("Upload PDF")
        pdf_file = st.file_uploader("Upload a PDF", type="pdf")
        if pdf_file:
            text = extract_text_from_pdf(pdf_file)
            chunks = chunk_text(text)
            embeddings = embed_text(chunks)
            faiss_index = create_faiss_index(embeddings)
            st.session_state.chunks = chunks
            st.session_state.faiss_index = faiss_index
            st.success("PDF processed successfully!")
        else:
            st.warning("Upload a PDF to start.")
            return
    
    # Chat history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Input section (Text input or mic button)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_input("Enter your question:", key="text_query", on_change=handle_query)
    with col2:
        audio_bytes = audio_recorder()

    if audio_bytes:
        user_query = speech_to_text(audio_bytes)
        
        # Retrieve relevant chunks and generate response
        relevant_chunks = retrieve_relevant_chunks(user_query, st.session_state.faiss_index, st.session_state.chunks)
        response_text = generate_response(user_query, relevant_chunks)
        
        # Convert response to voice if needed
        speech_file_path = text_to_speech(response_text)
        
        # Save to conversation history
        st.session_state.conversation_history.append({
            "user_query": user_query,
            "ai_response": response_text,
            "ai_response_voice": speech_file_path
        })
        
    # Display Chat History
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.conversation_history:
            st.markdown(f"**You:** {chat['user_query']}")
            st.markdown(f"**AI:** {chat['ai_response']}")
            if chat['ai_response_voice']:
                st.audio(chat['ai_response_voice'])

if __name__ == "__main__":
    main()
