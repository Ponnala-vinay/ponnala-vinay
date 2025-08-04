import os
from flask import Flask, request, jsonify
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Show if key is set
print("🔑 OPENAI_API_KEY =", "✅ SET" if os.getenv("OPENAI_API_KEY") else "❌ MISSING ❌")

# Initialize LLM
llm = ChatOpenAI(
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

# LangChain chain
question_chain = LLMChain(llm=llm, prompt=prompt_template)

# Flask app
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
