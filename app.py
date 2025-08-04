import os
from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import nest_asyncio
import threading

# Prevent Flask from crashing in Jupyter
nest_asyncio.apply()

# Set OpenRouter API key
os.environ["OPENAI_API_KEY"] = "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"

# Initialize the LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0.7
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template="""
You are a helpful assistant. Read the following text and generate 5 clear, concise, and relevant questions from it.

Text:
{input_text}

Questions:
"""
)

# Create the LangChain pipeline
question_chain = LLMChain(llm=llm, prompt=prompt_template)

# Set up Flask app
app = Flask(__name__)

@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    input_text = data.get("text")

    if not input_text:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    try:
        questions = question_chain.run({"input_text": input_text})
        return jsonify({"questions": questions.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

# Function to run Flask server in background
def run_app():
    app.run(port=5000, debug=False, use_reloader=False)

# Start Flask server in background thread
thread = threading.Thread(target=run_app)
thread.start()





import requests

response = requests.post(
    "http://127.0.0.1:5000/generate-questions",
    json={"text": """The process of evaporation involves the transformation of a liquid into a gas, usually under the influence of heat. 
            It plays a key role in the water cycle, contributing to the movement of water from the surface into the atmosphere."""}
)

print(response.json())
