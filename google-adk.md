ADK Agent Development: Local Models, RAG, Web, and Voice with Nextcloud
This document provides a comprehensive guide to setting up the Google Agent Development Kit (ADK) to work with local Large Language Models (LLMs), implement Retrieval Augmented Generation (RAG) and Web Access capabilities, and integrate Speech-to-Text (STT) and Text-to-Speech (TTS) for a complete conversational AI experience. This version also includes instructions for connecting to a local Nextcloud instance for document retrieval.
More info at: https://google.github.io/adk-docs/
Sample code: https://github.com/google/adk-samples
Table of Contents
1. ADK and Dependencies Installation
2. Integrating Your Custom Local LLM
• Using LiteLLM for OpenAI-Compatible Endpoints
• Other Local LLM Serving Options
3. Implementing Retrieval Augmented Generation (RAG)
• RAG Tool Definition
• RAG Backend Options
4. Adding Web Access (Search)
• Web Search Tool Definition
• Web Search API Options
5. Integrating Voice (STT & TTS)
• Speech-to-Text (STT) Options
• Text-to-Speech (TTS) Options
• Conceptual Voice Interaction Loop
6. Connecting to Nextcloud for Document Retrieval
• Nextcloud Integration Strategy
• Setup: Nextcloud Credentials and Folder Mapping
• Nextcloud Document Ingestion Tool
• Nextcloud Retrieval Tool (RAG Extension)
7. Running Your ADK Agent
• ADK CLI (adk run)
• ADK Web UI (adk web)
• Deployment Considerations
 
1. ADK and Dependencies Installation
The ADK is a Python-based framework, making installation straightforward using pip.
Prerequisites:
• Python 3.9+: Ensure you have a compatible Python version installed.
• Nextcloud Instance: A running Nextcloud instance (local or remote) with user credentials you can use.
Installation Steps:
1. 
Create a Project Directory:
Bash
Copy code
mkdir my_adk_agent
cd my_adk_agent
2. 
Create a Virtual Environment (Highly Recommended):
This isolates your project dependencies.
Bash
Copy code
python -m venv .venv
3. 
Activate the Virtual Environment:
• macOS / Linux:
Bash
Copy code
source .venv/bin/activate
• Windows (Command Prompt):
Bash
Copy code
.venv\Scripts\activate.bat
• Windows (PowerShell):
PowerShell
Copy code
.venv\Scripts\Activate.ps1
(Your terminal prompt should now show (.venv).)
4. 
Install ADK:
Bash
Copy code
pip install google-adk
5. 
Install Core Dependencies:
For the examples in this document, you'll need litellm for local LLM integration and fastapi/uvicorn for the ADK web UI, plus libraries for RAG, WebDAV for Nextcloud, and STT/TTS.
Bash
Copy code
pip install litellm fastapi uvicorn httpx pydantic openai
# For RAG Backend (example using ChromaDB and Sentence Transformers)
pip install chromadb sentence-transformers
# For Nextcloud WebDAV access
pip install webdav4[http] requests # requests is already common, but good to include
# For STT (Vosk)
pip install vosk SpeechRecognition sounddevice
# For TTS (Piper)
pip install piper-tts
# For Whisper (optional, but highly recommended STT)
pip install transformers accelerate soundfile librosa numpy torch
• litellm: Essential for unifying access to various LLMs, including your local one.
• fastapi, uvicorn, httpx, pydantic: Used by ADK's web UI and internal components.
• openai: The client library for OpenAI-compatible APIs, used by LiteLLM.
• chromadb, sentence-transformers: For a basic, easy-to-use RAG vector database and embedding model.
• webdav4[http]: Python client for WebDAV, crucial for interacting with Nextcloud's file system.
• vosk, SpeechRecognition, sounddevice: For local, offline speech-to-text.
• piper-tts: For fast, local, offline text-to-speech.
• transformers, accelerate, soundfile, librosa, numpy, torch: For the more accurate Whisper STT model.
6. 
Project Structure:
Create the basic ADK project structure:
Bash
Copy code
mkdir my_multi_agent
cd my_multi_agent
echo "from . import agent" > __init__.py
touch agent.py
touch tools.py
touch stt_tts_utils.py
touch .env
The .env file will store sensitive information like API keys and Nextcloud credentials.
2. Integrating Your Custom Local LLM
Your custom local model with an OpenAI-compatible API is an ideal candidate for integration via LiteLLM.
Using LiteLLM for OpenAI-Compatible Endpoints
LiteLLM provides a unified interface for over 100+ LLM providers, including those with OpenAI-compatible APIs.
my_multi_agent/agent.py (Initial Setup for Local LLM):
Python
Copy code
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# --- Configuration for your Custom Local LLM ---
# 1. Your Custom Model's API Base URL (e.g., from vLLM, text-generation-inference, or your own server)
CUSTOM_MODEL_BASE_URL = "http://localhost:8000/v1" # <--- IMPORTANT: Update with your actual endpoint!

# 2. The model name your custom endpoint uses (often "gpt-3.5-turbo", "llama-2", or a custom name)
CUSTOM_MODEL_NAME = "my-custom-llm-model" # <--- IMPORTANT: Update if your model has a specific name!

# 3. Dummy API Key for OpenAI client (often needed even if your local server doesn't validate it)
# It's best practice to put this in your .env file:
# OPENAI_API_KEY="sk-dummy-key"
# Then load it:
try:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-dummy-key")
except KeyError:
    print("Warning: OPENAI_API_KEY environment variable not set. Using dummy key. "
          "Ensure your local LLM doesn't require a valid key.")

# --- Initialize LiteLlm to point to your custom local model ---
# We tell LiteLlm to use the 'openai' provider, but override its base_url.
my_local_llm_instance = LiteLlm(
    model=f"openai/{CUSTOM_MODEL_NAME}",
    base_url=CUSTOM_MODEL_BASE_URL,
    # LiteLLM should automatically pick up OPENAI_API_KEY from os.environ
    # If not, you might need to pass it explicitly: api_key=OPENAI_API_KEY
)

# --- Define Your ADK Agent ---
main_agent = Agent(
    name="MyMultiCapableAgent",
    llm=my_local_llm_instance, # Assign your local LLM instance here
    description="A versatile agent capable of local RAG, web search, Nextcloud integration, and designed for voice interaction.",
    instruction=(
        "You are a helpful, knowledgeable, and polite AI assistant. "
        "Prioritize using your 'local_rag_tool' for specific document-based queries from your local knowledge base. "
        "Use the 'nextcloud_ingest_tool' to get a list of files on Nextcloud or 'nextcloud_retrieve_document_tool' to get content from Nextcloud. "
        "Use the 'web_search_tool' for general knowledge, current events, or information not found locally. "
        "Always provide concise, factual, and accurate answers. If a tool returns no results, state that. "
        "Maintain a conversational and friendly tone."
    ),
    tools=[] # We will add tools in subsequent sections
)

# The agent is defined. To run it, you'll use `adk run` or `adk web` from the command line
# in the parent directory (`my_adk_agent`).
my_multi_agent/.env:
OPENAI_API_KEY="sk-dummy-key" # Or a real API key if your custom model requires one
# Nextcloud Credentials (add these for the Nextcloud section)
NEXTCLOUD_URL="https://your-nextcloud-domain.com" # e.g., http://localhost/nextcloud
NEXTCLOUD_USERNAME="your_nextcloud_username"
NEXTCLOUD_PASSWORD="your_nextcloud_app_password" # Use an app password for security!
Other Local LLM Serving Options
If your custom model isn't yet served with an OpenAI-compatible API, here are common ways to expose it locally:
• Ollama: (Already discussed, but worth mentioning for its ease of use). Ollama provides a simple ollama serve command that exposes an OpenAI-compatible API. You can pull many open-source models (like Gemma, Llama 3, Mistral) and run them. 
• Installation: curl -fsSL https://ollama.com/install.sh | sh (Linux/macOS) or download for Windows.
• Usage: ollama pull <model_name>, then ollama serve. Your endpoint will be http://localhost:11434/v1.
• vLLM: A powerful and fast library for LLM inference, especially on GPUs. It can serve models with an OpenAI-compatible API. 
• Installation: pip install vllm. Requires a CUDA-enabled GPU.
• Usage: python -m vllm.entrypoints.api_server --model <model_path_or_hf_id> --served-model-name <your_model_name_for_api>. Default endpoint is http://localhost:8000/v1.
• text-generation-inference (TGI): Hugging Face's Rust-powered solution for serving LLMs, also with an OpenAI-compatible endpoint. Great for Hugging Face models. 
• Installation: Often via Docker.
• Usage: Refer to TGI documentation for Docker commands.
3. Implementing Retrieval Augmented Generation (RAG)
RAG allows your agent to fetch relevant information from a predefined knowledge base before generating a response, grounding its answers in specific data.
RAG Tool Definition
Create or update my_multi_agent/tools.py to house your custom tools.
my_multi_agent/tools.py:
Python
Copy code
from google.adk.agents.tools import Tool
from pydantic import BaseModel, Field
import os
import requests # For web search tool
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import warnings

# Suppress specific ChromaDB FutureWarning (optional)
warnings.filterwarnings("ignore", message="The sentence-transformers\
    /all-MiniLM-L6-v2 model is not natively supported by LiteLLM.", category=FutureWarning)


# --- RAG Backend Setup (ChromaDB and Sentence Transformers) ---
# Initialize ChromaDB client (will create 'chroma_db' directory if it doesn't exist)
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = PersistentClient(path=CHROMA_DB_PATH)

# Initialize embedding model (runs locally)
# Choose a model that balances performance and accuracy.
# 'all-MiniLM-L6-v2' is small and fast, 'all-MiniLM-L12-v2' is slightly larger/better.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Get or create a ChromaDB collection
# This is where your document chunks and embeddings will be stored.
# Ensure this collection exists before trying to add documents or query.
try:
    rag_collection = chroma_client.get_or_create_collection(name="my_rag_documents")
except Exception as e:
    print(f"Error getting/creating ChromaDB collection: {e}")
    rag_collection = None # Handle case where DB setup fails

def add_document_to_rag(doc_id: str, content: str, metadata: dict = None):
    """Adds a document chunk to the RAG vector database."""
    if rag_collection is None:
        print("RAG collection not initialized. Cannot add document.")
        return
    try:
        embedding = embedding_model.encode(content).tolist()
        rag_collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata if metadata else {}],
            ids=[doc_id]
        )
        print(f"Document '{doc_id}' added/updated in RAG DB.")
    except Exception as e:
        print(f"Failed to add document '{doc_id}' to RAG DB: {e}")

# --- RAG Tool ---
class RagToolInput(BaseModel):
    """Input schema for the local RAG tool."""
    query: str = Field(
        description="The specific question or keyword to search the local knowledge base for."
    )

def perform_local_rag(query: str) -> str:
    """
    Performs a search over the local RAG vector database and returns relevant text chunks.
    """
    print(f"--- RAG Tool Called with Query: '{query}' ---")
    if rag_collection is None:
        return "RAG system not available. Please ensure the knowledge base is set up."

    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = rag_collection.query(
            query_embeddings=[query_embedding],
            n_results=3, # Retrieve top 3 most relevant chunks
            include=['documents', 'metadatas']
        )

        retrieved_info = []
        if results and results['documents']:
            for i, doc_content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                source_info = f" (Source: {metadata.get('source', 'Unknown')})" if metadata else ""
                retrieved_info.append(f"Context{i+1}{source_info}:\n{doc_content}")
            return "\n\n".join(retrieved_info)
        else:
            return "No specific information found in the local knowledge base for that query."
    except Exception as e:
        return f"Error during RAG query: {e}. Please check your RAG setup."

# Register the RAG tool with ADK
local_rag_tool = Tool(
    name="local_rag_tool",
    description="Searches a local, predefined knowledge base for factual information. Use this for specific questions about known topics or internal documents.",
    input_model=RagToolInput,
    function=perform_local_rag,
)


# --- Web Search Tool ---
class WebSearchToolInput(BaseModel):
    """Input schema for the web search tool."""
    query: str = Field(
        description="The query to perform a general web search for. Be specific with keywords."
    )

def perform_web_search(query: str) -> str:
    """
    Performs a web search using a reliable search API and returns relevant snippets.
    This example uses the DuckDuckGo Instant Answer API for simplicity and no API key.
    For more robust use, integrate with a dedicated search API.
    """
    print(f"--- Web Search Tool Called with Query: '{query}' ---")
    try:
        # DuckDuckGo Instant Answer API for quick, no-API-key snippets
        url = f"https://api.duckduckgo.com/?q={query}&format=json&t=adk_agent_example"
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        result_snippets = []
        if 'Abstract' in data and data['Abstract']:
            result_snippets.append(f"Abstract: {data['Abstract']}")
        if 'Results' in data and data['Results']:
            for res in data['Results']:
                if 'Text' in res and 'FirstURL' in res:
                    result_snippets.append(f"Result: {res['Text']} ({res['FirstURL']})")
        if 'RelatedTopics' in data and data['RelatedTopics']:
            for topic in data['RelatedTopics']:
                if isinstance(topic, dict) and 'Text' in topic:
                    result_snippets.append(f"Related: {topic['Text']}")

        if result_snippets:
            return "\n".join(result_snippets[:3]) # Return top 3 relevant snippets
        else:
            return "Web search found no direct results for that query."
    except requests.exceptions.RequestException as e:
        return f"Error during web search: Could not connect or retrieve results. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred during web search: {e}"

web_search_tool = Tool(
    name="web_search_tool",
    description="Performs a general web search to find current information, news, or broad topics. Use this when information is not in the local knowledge base.",
    input_model=WebSearchToolInput,
    function=perform_web_search,
)
Update my_multi_agent/agent.py to include the RAG tool:
Python
Copy code
# ... (previous imports and LLM setup) ...
from .tools import local_rag_tool, web_search_tool # Import both tools

main_agent = Agent(
    name="MyMultiCapableAgent",
    llm=my_local_llm_instance,
    description="A versatile agent capable of local RAG, web search, Nextcloud integration, and designed for voice interaction.",
    instruction=(
        "You are a helpful, knowledgeable, and polite AI assistant. "
        "**Crucially, use the 'local_rag_tool' first for questions about specific, predefined topics or documents in the local knowledge base.** "
        "**Use the 'nextcloud_ingest_tool' to list files on Nextcloud or 'nextcloud_retrieve_document_tool' to get content from Nextcloud.** " # NEW
        "**Use the 'web_search_tool' for general knowledge, current events, or information not found locally.** "
        "Always provide concise, factual, and accurate answers. If a tool returns no results, state that clearly. "
        "Maintain a conversational and friendly tone."
    ),
    tools=[local_rag_tool, web_search_tool] # Add both tools here
)
RAG Backend Options
For a robust RAG system, your perform_local_rag function would interact with:
• Vector Databases (Vector Stores):
• ChromaDB: (Used in example) Lightweight, in-memory or persistent, easy to use. Good for local development.
• FAISS: Facebook AI Similarity Search, highly optimized for similarity search on large datasets.
• Milvus / Weaviate / Qdrant: More scalable, production-ready vector databases that can run locally via Docker.
• Embedding Models: You'll need an embedding model to convert your text documents (and queries) into numerical vectors. 
• Local Models: (Used in example, e.g., SentenceTransformer models). Many open-source embedding models (e.g., from Hugging Face like all-MiniLM-L6-v2) can be run locally via sentence-transformers or Ollama.
• OpenAI Embeddings / Google Embeddings API: If your local LLM server also offers an embedding API, you can use that. LiteLLM might help here too.
• Document Loaders: Libraries like Langchain or LlamaIndex provide robust ways to load data from various sources (PDFs, text files, websites, code) and split them into chunks for embedding. They also offer specific parsers for different file types (e.g., PyPDFLoader, DirectoryLoader, RecursiveCharacterTextSplitter for code).
4. Adding Web Access (Search)
Web access allows your agent to retrieve up-to-date information from the internet, making it more dynamic. (Tool definition already provided in Section 3).
Web Search Tool Definition
(See web_search_tool in my_multi_agent/tools.py above)
Web Search API Options
For production-grade web search, consider these options:
• Google Custom Search API (CSE): Offers programmatic access to Google Search results. Requires an API key and setting up a Custom Search Engine.
• SerpAPI / Serper.dev: Third-party APIs that scrape Google Search results, providing structured data. Often easier to integrate but come with costs.
• DuckDuckGo Instant Answer API: (Used in example) Simple, free, no API key needed, but returns limited snippets. Good for quick demonstrations.
• Brave Search API: A new, independent search engine offering an API.
• Custom Scraping: More complex and prone to breaking, but gives full control. Libraries like BeautifulSoup and requests or Playwright/Selenium for dynamic content.
5. Integrating Voice (STT & TTS)
Voice interaction requires converting speech to text (STT) for the agent's input and text to speech (TTS) for the agent's output. This typically happens outside the core ADK agent definition, in the application that interacts with the agent (e.g., your Python script or a custom UI).
Create my_multi_agent/stt_tts_utils.py and add the following content:
my_multi_agent/stt_tts_utils.py:
Python
Copy code
import os
import subprocess
import soundfile as sf
import sounddevice as sd
import torch
import numpy as np
import json
import speech_recognition as sr
from vosk import Model, KaldiRecognizer, SetLogLevel

# --- STT Setup ---
# Suppress Vosk logs (optional)
SetLogLevel(-1)

# IMPORTANT: Download Vosk model if you intend to use transcribe_audio_vosk_mic
# Download from: https://alphacephei.com/vosk/models (e.g., en-us-0.22)
# Unzip the downloaded model into your 'my_multi_agent' directory, e.g., 'my_multi_agent/vosk-model-en-us-0.22'
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model-en-us-0.22") # Update path if needed

# Whisper Setup (Load once for efficiency)
# Choose a smaller model like "base" or "small" for faster inference on local hardware.
# You need to install `transformers`, `accelerate`, `torch`, `soundfile`, `librosa`.
WHISPER_MODEL_NAME = "openai/whisper-tiny" # Consider "base", "small", "medium" for better accuracy
whisper_pipe = None
try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_NAME)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_NAME,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        chunk_length_s=30,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True # Useful for longer audio
    )
    print(f"Whisper model '{WHISPER_MODEL_NAME}' loaded on {device}.")
except ImportError:
    print("Whisper dependencies not fully installed. Skipping Whisper STT.")
except Exception as e:
    print(f"Could not load Whisper model: {e}. STT will rely on other methods if available.")

def transcribe_audio_vosk_mic() -> str:
    """Listens to microphone input and transcribes using Vosk."""
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"Vosk model not found at {VOSK_MODEL_PATH}. Please download it.")
        return ""

    r = sr.Recognizer()
    try:
        model = Model(VOSK_MODEL_PATH)
    except Exception as e:
        print(f"Error loading Vosk model: {e}")
        return ""

    with sr.Microphone(sample_rate=16000) as source:
        print("Listening (Vosk)...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=5) # Listen for up to 5 seconds of silence
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return ""

    try:
        rec = KaldiRecognizer(model, source.SAMPLE_RATE)
        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)

        if rec.AcceptWaveform(audio_data):
            result = json.loads(rec.Result())
            return result.get('text', '')
        else:
            partial_result = json.loads(rec.PartialResult())
            return partial_result.get('partial', '').strip()
    except sr.UnknownValueError:
        print("Vosk could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Vosk speech recognition service error: {e}")
        return ""
    except Exception as e:
        print(f"An error occurred during Vosk transcription: {e}")
        return ""

def transcribe_audio_whisper(audio_file_path: str) -> str:
    """Transcribes an audio file using the loaded Whisper model."""
    if not whisper_pipe:
        print("Whisper pipeline not initialized. Cannot perform STT.")
        return ""
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return ""
    try:
        # Load audio using soundfile (Whisper expects 16kHz, mono, float32)
        data, samplerate = sf.read(audio_file_path)
        if samplerate != 16000:
            import resampy # pip install resampy
            data = resampy.resample(data, samplerate, 16000)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.ndim > 1: # Convert to mono if stereo
            data = np.mean(data, axis=1)

        print(f"Transcribing {audio_file_path} with Whisper...")
        transcription = whisper_pipe(data)["text"]
        return transcription.strip()
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return ""

### Text-to-Speech (TTS) Options
# IMPORTANT: Configure these paths to your downloaded Piper files!
# Download Piper executable and voice models from: https://github.com/rhasspy/piper/releases
# Place them in a 'piper' sub-directory within your 'my_multi_agent' folder.
PIPER_EXE_PATH = os.path.join(os.path.dirname(__file__), "piper", "piper") # e.g., piper-linux-x64
PIPER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "piper", "en_US-lessac-medium.onnx") # Your chosen voice model
PIPER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "piper", "en_US-lessac-medium.onnx.json") # Corresponding config

def speak_text_piper(text: str, output_wav_path: str = "agent_response.wav") -> bool:
    """Generates speech from text using Piper and saves it to a WAV file."""
    if not (os.path.exists(PIPER_EXE_PATH) and os.path.exists(PIPER_MODEL_PATH) and os.path.exists(PIPER_CONFIG_PATH)):
        print(f"Piper files not found. Please check paths:")
        print(f"  Exe: {PIPER_EXE_PATH}")
        print(f"  Model: {PIPER_MODEL_PATH}")
        print(f"  Config: {PIPER_CONFIG_PATH}")
        return False

    try:
        command = [
            PIPER_EXE_PATH,
            "--model", PIPER_MODEL_PATH,
            "--config", PIPER_CONFIG_PATH,
            "--output_file", output_wav_path
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(input=text)
        process.wait()

        if process.returncode == 0:
            print(f"Speech generated and saved to {output_wav_path}")
            return True
        else:
            print(f"Piper error (Code {process.returncode}): {stderr.decode().strip()}")
            return False
    except FileNotFoundError:
        print(f"Error: Piper executable not found at '{PIPER_EXE_PATH}'.")
        return False
    except Exception as e:
        print(f"Error generating speech with Piper: {e}")
        return False

def play_wav_file(file_path: str):
    """Plays a WAV file using sounddevice."""
    if not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        return

    try:
        data, samplerate = sf.read(file_path)
        sd.play(data, samplerate)
        sd.wait() # Wait until the audio is finished playing
    except Exception as e:
        print(f"Error playing audio file {file_path}: {e}")

Conceptual Voice Interaction Loop
This loop would be part of a separate script or your main application that orchestrates the agent, STT, and TTS.
run_voice_agent.py (Example Orchestration Script):
Python
Copy code
import asyncio
import os
import sys

# Add your my_multi_agent directory to Python path if running from outside
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_multi_agent')))

# Assuming your agent is defined in my_multi_agent/agent.py
from my_multi_agent.agent import main_agent
# Assuming STT/TTS utilities are in my_multi_agent/stt_tts_utils.py
from my_multi_agent.stt_tts_utils import transcribe_audio_vosk_mic, transcribe_audio_whisper, speak_text_piper, play_wav_file

# For direct interaction with the agent's runner
from google.adk.agents import Runner, Session

async def run_voice_agent_loop():
    """
    Orchestrates the voice interaction loop with the ADK agent.
    Handles STT (mic input), sending to agent, and TTS (agent output).
    """
    print("Starting voice agent. Press Ctrl+C to exit.")

    # Initialize the ADK Runner for your agent
    agent_runner = Runner(main_agent)
    # Create a session for continuous conversation
    session = agent_runner.create_session(user_id="voice_user_123")

    # Speak a greeting to confirm startup
    speak_text_piper("Hello. I am your multi-capable AI assistant. How can I help you today?")
    play_wav_file("agent_response_temp.wav") # Reusing filename from stt_tts_utils

    while True:
        try:
            # --- 1. Speech-to-Text (User Input) ---
            print("\nListening for your command...")
            # Choose your STT method:
            # user_spoken_text = transcribe_audio_vosk_mic() # Using Vosk from microphone (Requires Vosk model)
            # OR record audio to file and then transcribe with Whisper:
            # (Requires setting up recording, e.g., using sounddevice to record to a WAV file)
            # For simplicity, let's just simulate reading from mic for now or user typing.
            # For a truly live Whisper mic, you'd need a more advanced streaming setup.
            user_input_mode = input("Speak (s) or type (t)? ").lower()
            if user_input_mode == 's':
                user_spoken_text = transcribe_audio_vosk_mic() # Or integrate live Whisper here
            else:
                user_spoken_text = input("You (type): ")

            if not user_spoken_text:
                print("No input detected or understood. Trying again...")
                continue
            if user_spoken_text.lower() in ['exit', 'quit', 'goodbye']:
                print("Exiting agent. Goodbye!")
                speak_text_piper("Goodbye!")
                play_wav_file("agent_response_temp.wav")
                break

            print(f"You said: {user_spoken_text}")

            # --- 2. Send to ADK Agent and Get Response ---
            print("Agent is processing...")
            # The `run_async` method handles the full agent reasoning process,
            # including tool calling, based on the agent's instructions.
            response_event = await agent_runner.run_async(user_spoken_text, session=session)
            agent_response_text = response_event.text

            if not agent_response_text:
                agent_response_text = "I couldn't generate a clear response. Can you rephrase?"

            print(f"Agent says: {agent_response_text}")

            # --- 3. Text-to-Speech (Agent Output) ---
            output_wav_file = os.path.join(os.path.dirname(__file__), "agent_response_temp.wav")
            if speak_text_piper(agent_response_text, output_wav_file):
                play_wav_file(output_wav_file)
                os.remove(output_wav_file) # Clean up the temp WAV file
            else:
                print("Failed to generate audio response.")

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Shutting down...")
            speak_text_piper("Shutting down. Goodbye!")
            play_wav_file(os.path.join(os.path.dirname(__file__), "agent_response_temp.wav"))
            break
        except Exception as e:
            print(f"An unexpected error occurred in the voice loop: {e}")
            import traceback
            traceback.print_exc() # For more detailed error info
            # Continue the loop or break based on desired behavior

if __name__ == "__main__":
    # Before running, ensure:
    # 1. Your custom local LLM server is running and accessible at CUSTOM_MODEL_BASE_URL.
    # 2. Vosk model is downloaded to my_multi_agent/vosk-model-en-us-0.22 (if using Vosk STT).
    # 3. Piper executable, model, and config are downloaded and paths configured correctly.
    # 4. NEXTCLOUD_URL, USERNAME, PASSWORD are set in your .env file.
    # 5. ChromaDB and embedding model are correctly set up (tools.py).

    # To run this script, navigate to the parent directory (`my_adk_agent`)
    # and execute: `python run_voice_agent.py`
    asyncio.run(run_voice_agent_loop())
6. Connecting to Nextcloud for Document Retrieval
Integrating with Nextcloud allows your agent to list, retrieve, and potentially ingest documents directly from your Nextcloud instance. We'll focus on two main ADK tools for this: one to list files and another to retrieve content for direct use or further RAG ingestion.
Nextcloud Integration Strategy
We'll use WebDAV, which is the primary way Nextcloud exposes its file system programmatically. The webdav4 Python library is suitable for this.
Flow for Nextcloud Integration:
1. Authentication: Use your Nextcloud URL, username, and an App Password (highly recommended over your main login password) for secure access.
2. Listing Files Tool: An ADK tool that queries Nextcloud via WebDAV to list files and folders. This helps the agent understand what documents are available.
3. Retrieving Document Content Tool: An ADK tool that fetches the actual content of a specific file from Nextcloud. This content can then be: 
• Directly presented to the LLM (for short, concise documents).
• Ingested into your local RAG vector database for more complex queries and long documents (if the agent determines this is necessary).
Setup: Nextcloud Credentials and Folder Mapping
1. 
Generate an App Password in Nextcloud:
• Log in to your Nextcloud instance.
• Go to your user settings (click your profile picture, then "Settings").
• Navigate to "Security" or "Mobile & Desktop clients".
• Under "App passwords" or "Create new app password", give it a name (e.g., "AI Agent") and click "Create new app password".
• Copy this generated password immediately! You won't see it again. This is what NEXTCLOUD_PASSWORD in your .env file should be.
2. 
Update my_multi_agent/.env:
# ... (existing entries) ...
NEXTCLOUD_URL="https://your-nextcloud-domain.com" # Example: http://localhost/nextcloud or https://your.domain.com/nextcloud
NEXTCLOUD_USERNAME="your_nextcloud_username"
NEXTCLOUD_PASSWORD="your_nextcloud_app_password" # PASTE THE APP PASSWORD HERE
3. 
Add os.getenv for Nextcloud credentials in my_multi_agent/tools.py:
Python
Copy code
# my_multi_agent/tools.py (add to top with other imports)
import os
from webdav4.client import Client as WebDAVClient
from urllib.parse import urljoin # For robust URL joining
# ... rest of your tools.py ...

# --- Nextcloud Configuration ---
NEXTCLOUD_URL = os.getenv("NEXTCLOUD_URL")
NEXTCLOUD_USERNAME = os.getenv("NEXTCLOUD_USERNAME")
NEXTCLOUD_PASSWORD = os.getenv("NEXTCLOUD_PASSWORD")

# WebDAV base URL is usually the Nextcloud URL + '/remote.php/dav/files/<username>/'
# Adjust this path based on your Nextcloud setup and desired root folder.
# For a direct user's files root:
NEXTCLOUD_WEBDAV_BASE_URL = urljoin(NEXTCLOUD_URL, f'/remote.php/dav/files/{NEXTCLOUD_USERNAME}/')

nextcloud_client = None
if NEXTCLOUD_URL and NEXTCLOUD_USERNAME and NEXTCLOUD_PASSWORD:
    try:
        nextcloud_client = WebDAVClient(
            base_url=NEXTCLOUD_WEBDAV_BASE_URL,
            username=NEXTCLOUD_USERNAME,
            password=NEXTCLOUD_PASSWORD
        )
        # Test connection (optional, good for debugging setup)
        # nextcloud_client.ls('/')
        print(f"Connected to Nextcloud WebDAV at {NEXTCLOUD_WEBDAV_BASE_URL}")
    except Exception as e:
        print(f"Error connecting to Nextcloud: {e}. Please check .env settings and Nextcloud availability.")
        nextcloud_client = None
else:
    print("Nextcloud credentials not fully set in .env. Nextcloud tools will not be available.")

Nextcloud Document Ingestion Tool
This tool helps the agent list files within Nextcloud. We'll call it nextcloud_ingest_tool as it's the first step to "ingesting" knowledge from Nextcloud into the agent's awareness.
Add to my_multi_agent/tools.py:
Python
Copy code
# ... (existing imports and tools, and Nextcloud setup) ...

# --- Nextcloud Ingestion Tool (for listing files/folders) ---
class NextcloudIngestToolInput(BaseModel):
    """Input schema for listing Nextcloud contents."""
    path: str = Field(
        description="The path within Nextcloud to list (e.g., '/', '/Documents/', '/MyCode/'). Must start and end with '/'.",
        default="/"
    )

def list_nextcloud_contents(path: str = "/") -> str:
    """Lists files and folders in a specified Nextcloud path."""
    print(f"--- Nextcloud Ingest Tool Called to list: '{path}' ---")
    if nextcloud_client is None:
        return "Nextcloud connection not established. Please check credentials."

    try:
        # Ensure path starts and ends with '/' for WebDAV
        if not path.startswith('/'):
            path = '/' + path
        if not path.endswith('/'):
            path = path + '/'

        items = nextcloud_client.ls(path)
        if not items:
            return f"No items found in '{path}' on Nextcloud or path does not exist."

        # Filter out the base path itself if it's listed (WebDAV behavior)
        filtered_items = [item for item in items if item.name != '' and item.name != '.']

        # Format output
        output = []
        for item in filtered_items:
            item_type = "Folder" if item.is_dir else "File"
            item_size = f"({item.size_text})" if not item.is_dir else ""
            output.append(f"- {item_type}: {item.name} {item_size}")

        return f"Contents of '{path}' on Nextcloud:\n" + "\n".join(output)
    except Exception as e:
        return f"Error listing Nextcloud contents at '{path}': {e}"

nextcloud_ingest_tool = Tool(
    name="nextcloud_ingest_tool",
    description="Lists files and folders available in a specified path on Nextcloud. Use this to understand what documents are present.",
    input_model=NextcloudIngestToolInput,
    function=list_nextcloud_contents,
)
Nextcloud Retrieval Tool (RAG Extension)
This tool will fetch the actual content of a document from Nextcloud. The agent can use this content directly or decide to pass it to the RAG system for deeper indexing and querying.
Add to my_multi_agent/tools.py:
Python
Copy code
# ... (existing imports and tools, and Nextcloud setup) ...

# --- Nextcloud Retrieval Tool ---
class NextcloudRetrieveDocumentInput(BaseModel):
    """Input schema for retrieving a document from Nextcloud."""
    file_path: str = Field(
        description="The full path to the file on Nextcloud (e.g., '/Documents/report.pdf', '/MyCode/main.py'). Must start with '/'.",
    )

def retrieve_nextcloud_document(file_path: str) -> str:
    """
    Retrieves the content of a specific document from Nextcloud.
    It returns the content as a string. For binary files like PDFs,
    you'd need to add a PDF parser here.
    """
    print(f"--- Nextcloud Retrieve Document Tool Called for: '{file_path}' ---")
    if nextcloud_client is None:
        return "Nextcloud connection not established. Please check credentials."

    if not file_path.startswith('/'):
        file_path = '/' + file_path

    try:
        # Check if it's a directory
        try:
            item_info = nextcloud_client.stat(file_path)
            if item_info.is_dir:
                return f"Error: '{file_path}' is a directory, not a file. Use 'nextcloud_ingest_tool' to list its contents."
        except Exception: # File might not exist yet, or other error
             pass # Will be caught by read_text later

        # For text-based files (txt, md, py, json, etc.)
        content = nextcloud_client.read_text(file_path)
        # You might want to truncate very long documents before passing to LLM
        # Or, trigger ingestion into your local RAG DB here if the document is too large
        if len(content) > 4000: # Example limit
            # This is where the agent could decide to push to RAG DB
            # add_document_to_rag(doc_id=file_path, content=content, metadata={"source": "Nextcloud", "path": file_path})
            # return f"Document '{file_path}' is too large to display directly, but its content has been indexed for future RAG queries."
            return f"Retrieved content for '{file_path}' (truncated to 4000 chars):\n" + content[:4000] + "..."
        return f"Content of '{file_path}':\n```\n{content}\n```"
    except Exception as e:
        return f"Error retrieving document '{file_path}' from Nextcloud: {e}. Check path and permissions."

nextcloud_retrieve_document_tool = Tool(
    name="nextcloud_retrieve_document_tool",
    description="Retrieves the full text content of a specified file from Nextcloud. Use this when you need the exact content of a document.",
    input_model=NextcloudRetrieveDocumentInput,
    function=retrieve_nextcloud_document,
)
Update my_multi_agent/agent.py to include the Nextcloud tools:
Python
Copy code
# ... (previous imports and LLM setup) ...
from .tools import (
    local_rag_tool,
    web_search_tool,
    nextcloud_ingest_tool,          # NEW
    nextcloud_retrieve_document_tool # NEW
)

main_agent = Agent(
    name="MyMultiCapableAgent",
    llm=my_local_llm_instance,
    description="A versatile agent capable of local RAG, web search, Nextcloud integration, and designed for voice interaction.",
    instruction=(
        "You are a helpful, knowledgeable, and polite AI assistant. "
        "**For specific document-based queries, first use the 'local_rag_tool'.** "
        "**To explore files on Nextcloud, use 'nextcloud_ingest_tool' (e.g., to list a folder's contents).** "
        "**To get the content of a specific file from Nextcloud, use 'nextcloud_retrieve_document_tool'.** "
        "**For general knowledge, current events, or information not found locally or on Nextcloud, use the 'web_search_tool'.** "
        "Always provide concise, factual, and accurate answers. If a tool returns no results, state that clearly. "
        "Maintain a conversational and friendly tone."
    ),
    tools=[
        local_rag_tool,
        web_search_tool,
        nextcloud_ingest_tool,          # NEW
        nextcloud_retrieve_document_tool # NEW
    ]
)
Nextcloud Document Ingestion (for RAG) Tool
The retrieve_nextcloud_document tool primarily fetches content. For longer documents or a persistent Nextcloud-backed RAG, you'd extend this:
• Trigger RAG Ingestion: When retrieve_nextcloud_document gets a large file, instead of just returning truncated text, it could call add_document_to_rag (defined in tools.py) to chunk and embed the Nextcloud document into your local ChromaDB. The LLM would then be informed that the document is now indexed for future RAG queries.
• Background Sync: For a continuously updated Nextcloud RAG, you'd likely have a separate, long-running script that periodically scans your Nextcloud folders, identifies new/modified files, retrieves them, and ingests them into your ChromaDB. This happens completely independently of the ADK agent's live interaction.
This setup gives your agent powerful capabilities: it can answer questions based on its local RAG, perform web searches, and now, dynamically browse and retrieve documents from your Nextcloud instance.
 