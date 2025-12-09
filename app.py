from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# ---------- Load environment variables ----------
load_dotenv()  # take environment variables from .env
api_key = os.getenv("GOOGLE_API_KEY")

# ---------- 1. Configure API Key ----------
genai.configure(api_key=api_key)

# ---------- 2. Load LLM Model ----------
model = genai.GenerativeModel("models/gemini-flash-lite-latest")

# ---------- 3. Load Synthetic Dataset from JSON ----------
with open("dataset.json", "r") as f:
    synthetic_data = json.load(f)

# ---------- 4. Build System Prompt ----------
system_prompt = "You are a Customer Support Chatbot for an E-commerce Store.\n"
system_prompt += "Use the following examples to answer user queries:\n"
for item in synthetic_data:
    system_prompt += f"- Q: {item['question']} -> A: {item['answer']}\n"
system_prompt += "\nAnswer politely and clearly any new user question based on these examples."

# ---------- 5. Chat Function ----------
def ask_llm(message):
    response = model.generate_content(system_prompt + "\nUser: " + message)
    return response.text

# ---------- 6. Flask App ----------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_reply = ask_llm(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
