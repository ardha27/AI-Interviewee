import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the API key from the environment variable
XI_API_KEY = os.getenv('elevenlabs_api')

# API endpoint URL
url = "https://api.elevenlabs.io/v1/voices"

# Set up headers for the HTTP request
headers = {
    "Accept": "application/json",
    "xi-api-key": XI_API_KEY,
    "Content-Type": "application/json"
}

# Send GET request to the API
response = requests.get(url, headers=headers)

# Parse the JSON response
data = response.json()

# Iterate over each voice in the response and print its name and ID
for voice in data['voices']:
    print(f"{voice['name']}; {voice['voice_id']}")
