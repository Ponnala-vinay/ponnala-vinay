from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Read API key from environment variables
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "mistralai/mixtral-8x7b-instruct"
URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter(sentence):
    """Send a request to OpenRouter API to generate follow-up questions using Mistral."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    You are a helpful assistant. Based on the following text, generate 5 clear, concise, and relevant follow-up questions.

    Text:
    "{sentence}"

    Questions:
    """

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7
    }

    try:
        response = requests.post(URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        return f"Request Error: {str(e)}"
    except KeyError:
        return "Error: Unexpected API response format."

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """API endpoint to generate questions from given text using Mistral LLM."""
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({"error": "Please provide 'sentence' in JSON body."}), 400

    sentence = data['sentence']
    questions = call_openrouter(sentence)

    if questions.startswith("Error:") or questions.startswith("Request Error:"):
        return jsonify({"error": questions}), 500

    return jsonify({"questions": questions})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
