from src.components.RAG import EmbeddingProcess
from src.components.Agent import Chatbot
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)  

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Chatbot API! Use POST /chat to interact with the chatbot."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_question = data.get("question", "")
    if not user_question:
        return jsonify({"error": "Question is required"}), 400
    try:
        response = app1.invoke({"messages": [user_question]})
        return jsonify({"response": response['messages'][-1].content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    vector = EmbeddingProcess()
    bot = Chatbot(vector.retriever)
    app1 = bot()
    app.run(debug=True, use_reloader=False)