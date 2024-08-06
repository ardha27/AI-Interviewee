# AI-Powered Interview Assistant

This project is an AI-powered interview assistant that uses speech recognition, natural language processing, and text-to-speech technologies to simulate an interview experience.

## Features

- Real-time speech detection and recording
- Audio transcription using Whisper AI
- Conversation management with GPT-4o-mini
- Text-to-speech response generation using ElevenLabs API
- CV content extraction for context-aware responses

## Prerequisites

- Python 3.7+
- OpenAI API key
- ElevenLabs API key
- PDF file containing your CV

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your API keys:
   ```
   chatgpt_api=your_openai_api_key
   elevenlabs_api=your_elevenlabs_api_key
   ```

## Usage

1. Place your CV PDF file in the project directory
2. Update the `CV_PATH` variable in the script with your CV file name
3. Run the script:
   ```
   python tes.py
   ```
4. Start speaking when prompted. The assistant will transcribe your speech, generate a response, and play it back as audio.

## Configuration

- Adjust `THRESHOLD` in the script to fine-tune speech detection sensitivity
- Modify `SILENCE_DURATION` and `MIN_DURATION` to change recording behavior
- Update `voice_model_id` to use a different ElevenLabs voice

## Note

This project uses various AI models and APIs, which may have associated costs. Please review the pricing and usage terms of OpenAI, ElevenLabs, and other services used in this project.
