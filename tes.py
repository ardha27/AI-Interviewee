import pyaudio
import wave
import time
import numpy as np
from openai import OpenAI
import os
import whisper
import PyPDF2
import requests
import json
import playsound
import tiktoken


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the ChatGPT API key from the environment variable
chatgpt_api_key = os.getenv('chatgpt_api')
XI_API_KEY = os.getenv('elevenlabs_api')

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 500  # Adjust this value based on your environment
SILENCE_DURATION = 1  # 1 second of silence
MIN_DURATION = 3  # Minimum duration of speech to save the recording
OUTPUT_DIR = 'recordings'
MAX_CONTEXT_LENGTH = 128000  # Maximum context length for GPT-4o-mini (128k)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize conversation history
conversation_history = []

def is_silent(data):
    """ Returns 'True' if below the 'silent' threshold """
    return max(data) < THRESHOLD

def record_to_file(path, frames):
    """ Save the recorded data to a wav file """
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(path):
    """ Transcribe the recorded audio using Whisper AI """
    model = whisper.load_model("large-v3")
    result = model.transcribe(path)
    return result['text']

def extract_cv_content(pdf_path):
    """ Extract text content from the CV PDF """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        cv_content = ""
        for page in reader.pages:
            cv_content += page.extract_text()
    return cv_content

def num_tokens_from_messages(messages, model="gpt-4o-mini"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def chatgpt_response(transcription, cv_content):
    client = OpenAI(api_key=chatgpt_api_key)
    
    # Add user's message to conversation history
    conversation_history.append({"role": "user", "content": transcription})
    
    # Prepare messages for API call
    messages = [
        {"role": "system", "content": f"Anda adalah seorang yang sedang diwawancarai. Jawablah pertanyaan secara singkat, padat, dan jelas. Berikan jawaban panjang lebar hanya jika diminta. Gunakan informasi CV berikut untuk menjawab: {cv_content}"},
    ]
    
    # Add conversation history, ensuring total token count doesn't exceed MAX_CONTEXT_LENGTH
    current_tokens = num_tokens_from_messages([messages[0]])

    for message in reversed(conversation_history):
        message_tokens = num_tokens_from_messages([message])
        if current_tokens + message_tokens <= MAX_CONTEXT_LENGTH:
            messages.insert(1, message)
            current_tokens += message_tokens
        else:
            break

    # Remove oldest messages if we still exceed the limit
    while current_tokens > MAX_CONTEXT_LENGTH and len(messages) > 2:
        removed_message = messages.pop(1)
        current_tokens -= num_tokens_from_messages([removed_message])
        print(f"Removed message. New token count: {current_tokens}")
    
    print(f"Final token count before API call: {current_tokens}")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    assistant_response = response.choices[0].message.content
    
    # Add assistant's response to conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    return assistant_response

def text_to_speech(text, voice_id):
    CHUNK_SIZE = 1024
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "response.mp3")

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    headers = {
        "Accept": "application/json",
        "xi-api-key": XI_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    response = requests.post(tts_url, headers=headers, json=data, stream=True)

    if response.ok:
        with open(OUTPUT_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        print("Audio response saved successfully.")
    else:
        print("Error in text-to-speech conversion:", response.text)

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Extracting CV content...")
    cv_content = extract_cv_content(CV_PATH)
    print("CV content extracted.")

    frames = []
    silent_chunks = 0
    recording = False
    start_time = None

    print("Listening...")

    try:
        while True:
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)
            silent = is_silent(audio_data)

            if not silent:
                if not recording:
                    print("Speech detected, starting recording...")
                    recording = True
                    start_time = time.time()
                frames.append(data)
                silent_chunks = 0
            else:
                if recording:
                    frames.append(data)
                    silent_chunks += 1
                    if silent_chunks > int(SILENCE_DURATION * RATE / CHUNK):
                        duration = time.time() - start_time
                        if duration > MIN_DURATION:
                            filename = os.path.join(OUTPUT_DIR, f"recording.wav")
                            print(f"Saving recording to {filename}")
                            record_to_file(filename, frames)
                            print("Transcribing audio...")
                            transcription = transcribe_audio(filename)
                            print(f"Transcription:\n{transcription}")
                            response = chatgpt_response(transcription, cv_content)
                            print(f"Response:\n{response}")
                            text_to_speech(response, voice_model_id)
                            print("Playing generated audio...")
                            audio_file = "recordings/response.mp3"  # Assuming this is the file name used in text_to_speech function
                            max_attempts = 5
                            attempt = 0
                            while attempt < max_attempts:
                                if os.path.exists(audio_file):
                                    try:
                                        # Wait for a short time to ensure the file is ready
                                        time.sleep(0.5)
                                        playsound.playsound(audio_file)
                                        break
                                    except Exception as e:
                                        print(f"Error playing audio (attempt {attempt + 1}): {str(e)}")
                                        attempt += 1
                                        time.sleep(1)  # Wait before retrying
                                else:
                                    print(f"Audio file {audio_file} not found. Waiting...")
                                    attempt += 1
                                    time.sleep(1)  # Wait before checking again
                            
                            if attempt == max_attempts:
                                print(f"Failed to play audio after {max_attempts} attempts.")
                        else:
                            print(f"Recording discarded, duration was only {duration:.2f} seconds")
                        frames = []
                        recording = False
                        silent_chunks = 0
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    CV_PATH = 'CV2023.pdf'  # Add the path to your CV PDF file
    voice_model_id = "S8mmuuHWcaCJboTOgjhr"
    record_audio()
