"""This chatbot has ability to chat using open AI LLMs and
if asked for weather for a specific location it can tell using weather api

 additional features are:
 1. saving response in a json file
 2. Loading past chats and using it for context based conversation
 3. Logging of error in a log file
 """

from openai import OpenAI
from datetime import datetime
import requests
import dotenv
import json
import os
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Load environment variables from .env file
dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

# Configuration
HISTORY_FILE = "chat_history.json"

class ChatBot:
    def __init__(self, api_key,weather_api_key, model='gpt-4o',temperature=0.5):
        self.api_key = api_key
        self.weather_api_key = weather_api_key
        self.model = model
        self.temperature = temperature
        self.chatHistory = []
        self.load_history()

    def chat(self, user_message,include_context: bool = True, max_context: int = 3):
        messages=[]
        # Check for weather command pattern
        if self.is_weather_query(user_message) and self.weather_api_key:
            return self.handle_weather_query(user_message)

        if include_context and self.chatHistory:
            for exchange in self.chatHistory[-max_context:]:
                messages.append({'role':'user','content':exchange['Prompt']})
                messages.append({'role':'assistant','content':exchange['Response']})

        print(f'{messages[-max_context:]}')
        print('='*10)
        messages.append({'role':'user','content':user_message})

        try:
            client = OpenAI(api_key=self.api_key)
            print(f"Waiting for response")
            response = client.chat.completions.create(
                model= self.model,
                messages= messages,
                temperature= self.temperature
            )
            response_content = response.choices[0].message.content
            self.save_chats(user_message=user_message, response=response_content)
            return response_content
        except Exception as e:
            return f"An error occurred: {e}"

    def is_weather_query(self,user_message):
        """Check if message contains weather-related keywords"""
        weather_keywords = ['weather','temp','climate', 'temperature', 'forecast', 'rain', 'snow']
        return any(keyword in user_message.lower() for keyword in weather_keywords)

    def extract_locations_with_nltk(self, text: str):
        """Extract location names using NLTK's named entity recognition"""
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)

        locations = []
        current_chunk = []

        # Words to exclude from location detection
        exclude_words = {'weather', 'temperature', 'temp', 'forecast', 'rain', 'snow', 'climate'}

        for entity in entities:
            if isinstance(entity, Tree):
                # Handle multi-word entities
                if entity.label() in ['GPE', 'LOCATION']:
                    location = " ".join([token for token, pos in entity.leaves()])
                    if location.lower() not in exclude_words:
                        locations.append(location)
            else:
                word, tag = entity
                # Handle consecutive proper nouns
                if tag == 'NNP' and word.istitle() and word.lower() not in exclude_words:
                    current_chunk.append(word)
                else:
                    if current_chunk:
                        potential_location = " ".join(current_chunk)
                        if potential_location.lower() not in exclude_words:
                            locations.append(potential_location)
                        current_chunk = []

        # Add any remaining chunks
        if current_chunk:
            potential_location = " ".join(current_chunk)
            if potential_location.lower() not in exclude_words:
                locations.append(potential_location)

        # Merge adjacent location words (e.g., "New" + "York")
        merged_locations = []
        i = 0
        while i < len(locations):
            if i + 1 < len(locations) and locations[i + 1].istitle():
                merged = f"{locations[i]} {locations[i + 1]}"
                merged_locations.append(merged)
                i += 2
            else:
                merged_locations.append(locations[i])
                i += 1

        return merged_locations[0]

    def handle_weather_query(self,user_message):
        try:
            # Extract location from message (simple implementation)
            location = self.extract_locations_with_nltk(user_message)

            # Get weather data
            weather_data = self.get_weather_data(location.lower())

            if weather_data:
                weather_response = (
                    f"Weather in {location.title()}:\n"
                    f"Temperature: {weather_data['temp']}°C\n"
                    f"Wind: {weather_data['wind_speed']} km/h\n"
                )
                self.save_chats(user_message, weather_response)
                return weather_response
            return "Couldn't fetch weather data. Please try again later."
        except Exception as e:
            self.log_error(f"Weather API error: {str(e)}")
            return "Sorry, I couldn't process the weather request."

    def get_weather_data(self,location='london'):
        try:
            base_url = "http://api.weatherapi.com/v1"
            curr_weather = "/current.json"
            params = {
                "key": self.weather_api_key,
                "q": location,
                "aqi": "yes"
            }
            weather_response = requests.get(base_url + curr_weather, params=params)
            weather_response = weather_response.json()
            return {
                'temp': weather_response['current']['temp_c'],
                'wind_speed': weather_response['current']['wind_kph'],
                'air_quality': weather_response['current']['air_quality']['pm2_5']
            }
        except Exception as e:
            self.log_error(f"Weather API call error: {str(e)}")
            return None

    def load_history(self):
        try:
            with open('chat_history.json', 'r', encoding="utf-8") as f:
                self.chatHistory = json.load(f)
        except Exception as e:
            print(f"Error loading history from JSON: {e}")
            with open('errorLog.txt', 'w', encoding="utf-8") as f:
                current_timestamp = datetime.now().isoformat()
                f.write(f'Timestamp: {current_timestamp} Error: {e}''\n')

    def save_chats(self, user_message, response):
        current_timestamp = datetime.now().isoformat()
        entry = {"Timestamp": current_timestamp,
                 "Prompt": user_message,
                 "Response": response
        }
        self.chatHistory.append(entry)
        try:
            with open('chat_history.json','w',encoding="utf-8") as f:
                json.dump(self.chatHistory,f,indent=4)
        except Exception as e:
            print(f"Error saving to JSON: {e}")

    def log_error(self,logMessage):
        with open('errorLog.txt', 'w', encoding="utf-8") as f:
            current_timestamp = datetime.now().isoformat()
            f.write(f'Timestamp: {current_timestamp} {logMessage}\n')



# Test cases
test_cases = [
    "What Is The Weather In Phoenix?",
    "What good way to learn AI that can be used in scale ",
    "Tell me about the climate in London",
    "Give a list of 10 companies from the locations discussed above"
]


try:
    for user_message in test_cases:
        # bot = ChatBot(api_key, weather_api_key, model= 'gpt-4.1-mini',temperature=0.7)
        bot = ChatBot(api_key, weather_api_key, model= 'gpt-3.5-turbo',temperature=0.7)
        response_content = bot.chat(user_message)
        print(response_content)
except Exception as e:
    try:
        with open('errorLog.txt', 'w', encoding="utf-8") as f:
            current_timestamp = datetime.now().isoformat()
            f.write(f'Timestamp: {current_timestamp} Error: {e}''\n')
    except Exception as e:
        print(f"Error saving to JSON: {e}")


