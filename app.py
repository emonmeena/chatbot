from flask import Flask, render_template, request, jsonify
from biobot import agent_executor

from langchain.agents.self_ask_with_search.output_parser import SelfAskOutputParser
from flask_cors import CORS




app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['user_message']
    response = agent_executor(user_message)
    return jsonify({'response': response['output']})

if __name__ == '__main__':
    app.run()
