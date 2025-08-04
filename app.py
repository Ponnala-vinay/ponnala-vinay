import os
from flask import Flask, request, jsonify
from langchain_community.chat_models import ChatOpenRouter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ✅ Read API key from environment variable
api_key = os.getenv("OPENROUTER_API_KEY")
print("🔑 OPENROUTER_API_KEY =", "✅ SET" if api_key else "❌ MISSING ❌")

# ✅ Set up LLM via OpenRouter
llm = ChatOpenRouter(
    model="mistralai/mixtral-8x7b-instruct",
    api_key=api_key,
    temperature=0.7,
)

# ✅ Prompt template
prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template="""
You are a helpful assistant. Read the following text and generate 5 clear, concise, and relevant questions from it.

Text:
{input_text}

Questions:
"""
)

# ✅ LangChain pipeline
question_chain = LLMChain(llm=llm, prompt=prompt_template)

# ✅ Flask API
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
