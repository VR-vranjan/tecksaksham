import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Function to fetch news from The Guardian API (No API Key Required)
def fetch_news():
    url = "https://content.guardianapis.com/search?order-by=newest&show-fields=headline"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("response", {}).get("results", [])
        return random.choice(articles)["webTitle"] if articles else "No news available right now."
    return "Failed to fetch news."

# Function to fetch stock market data from Twelve Data API (No API Key Required)
def fetch_trade():
    url = "https://api.twelvedata.com/time_series?symbol=AAPL&interval=1day&outputsize=1&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "values" in data:
            return f"Apple Stock Price: ${data['values'][0]['close']}"
    return "Failed to fetch trade data."

# Function to fetch weather data from wttr.in
def fetch_weather():
    url = "https://wttr.in/?format=%C+%t"
    response = requests.get(url)
    return response.text if response.status_code == 200 else "Failed to fetch weather."

# Load intents from JSON
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = input_text.lower()
    if "news" in input_text:
        return fetch_news()
    elif "trade" in input_text or "market" in input_text:
        return fetch_trade()
    elif "weather" in input_text:
        return fetch_weather()
    else:
        input_text = vectorizer.transform([input_text])
        tag = clf.predict(input_text)[0]
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "Sorry, I don't understand."

counter = 0

def main():
    global counter
    st.title("Chatbot: News, Trade & Weather")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Ask about news, trade, or weather updates!")
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("This chatbot provides real-time news, trade, and weather updates using free APIs without requiring an API key.")

if __name__ == '__main__':
    main()
