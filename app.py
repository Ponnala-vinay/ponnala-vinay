import os
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


api_key = os.getenv("OPENROUTER_API_KEY")
print("🔑 OPENROUTER_API_KEY =", "✅ SET" if api_key else "❌ MISSING ❌")

llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template="""
You are a helpful assistant. Read the following text and generate 5 clear, concise, and relevant questions from it.

Text:
{input_text}

Questions:
"""
)

question_chain = LLMChain(llm=llm, prompt=prompt_template)

app = Flask(__name__)

@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    try:
        questions = question_chain.run({"input_text": text})
        return jsonify({"questions": questions.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

