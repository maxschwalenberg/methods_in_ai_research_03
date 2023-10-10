from flask import Flask, render_template, request, jsonify


app = Flask(__name__)


@app.route("/")
def chatbot():
    return render_template("chatbot.html")


@app.route("/api/user_input", methods=["POST"])
def api_return_response():
    data = request.get_json()

    return jsonify({"response": "comes from the server"}), 200
