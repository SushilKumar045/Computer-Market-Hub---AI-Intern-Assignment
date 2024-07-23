!pip install langchain transformers flask

!pip install --upgrade PyPDF2
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file) # Use PdfReader instead of PdfFileReader
        text = ''
        for page_num in range(len(reader.pages)): # Use reader.pages to get the number of pages
            text += reader.pages[page_num].extract_text()
    return text

pdf_path = '/content/Corpus.pdf'  # Update this path
corpus_text = extract_text_from_pdf(pdf_path)

import json

with open('/content/Sample Question Answers.json', 'r') as file:  # Update this path
    sample_qas = json.load(file)

from transformers import pipeline

model_name = "gpt-3.5-turbo"  # You can use a smaller model to reduce cost
nlp = pipeline("text-generation", model=model_name)

from langchain import LLMChain, OpenAI
from langchain.prompts import ChatPromptTemplate, ChatPrompt

class WineChatbot:
    def __init__(self, corpus_text, sample_qas):
        self.corpus_text = corpus_text
        self.sample_qas = sample_qas
        self.chain = LLMChain(
            llm=OpenAI(model=model_name),
            prompt=ChatPromptTemplate("Answer based on the corpus: {corpus}.\n\n{question}")
        )

    def get_answer(self, question):
        for qa in self.sample_qas:
            if question.lower() in qa['question'].lower():
                return qa['answer']
        response = self.chain.run({
            'corpus': self.corpus_text,
            'question': question
        })
        return response or "Please contact the business directly for more information."

chatbot = WineChatbot(corpus_text, sample_qas)

from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
        <!doctype html>
        <html>
        <head><title>Wine Chatbot</title></head>
        <body>
            <h1>Wine Chatbot</h1>
            <div id="chat-box"></div>
            <input type="text" id="user-input" placeholder="Ask a question..." />
            <button onclick="sendMessage()">Send</button>
            <script>
                function sendMessage() {
                    var userInput = document.getElementById('user-input').value;
                    fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({question: userInput}),
                    })
                    .then(response => response.json())
                    .then(data => {
                        var chatBox = document.getElementById('chat-box');
                        chatBox.innerHTML += '<p>User: ' + userInput + '</p>';
                        chatBox.innerHTML += '<p>Bot: ' + data.answer + '</p>';
                    });
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']
    answer = chatbot.get_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run()

!python /path/to/your_script.py  # Update this path

