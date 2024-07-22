
# AskMe : AI Assistant

AskMe is an advanced AI assistant designed to transform the way enterprises interact with their vast repositories of documents. Utilizing cutting-edge natural language processing (NLP) and state-of-the-art AI technologies, AskMe provides instant, accurate, and context-aware responses to user queries, ensuring efficient information retrieval and enhanced productivity.

# Cutting-Edge Technology

#### Langchain Integration
Leveraging Langchain for robust document processing and query handling, AskMe ensures reliable and efficient performance.
#### Google Generative AI Embeddings
Enhancing document embeddings with Google’s generative AI technology, AskMe provides superior accuracy and relevance in responses.
#### Natural Language Processing
Utilizing state-of-the-art NLP models, AskMe can understand and interpret complex queries, ensuring precise information retrieval.
#### AI-Powered Search
Powered by advanced AI algorithms, AskMe delivers highly accurate search results, enhancing the efficiency of information discovery.
#### Vector Embeddings
Employing vector embeddings for document representation, AskMe ensures relevant and contextually appropriate responses.
#### Streamlit Interface
With a user-friendly interface built on Streamlit, AskMe offers an intuitive and accessible user experience.

## Getting API keys

 - [Get Google API key](https://aistudio.google.com/app/apikey)
 - [Get Groq API Key](https://console.groq.com/keys)






## Installation

Install requirements with pip

```bash
  pip install -r requirements.txt
```
    

## SetUp

* Create a .env File

Create a .env file in the root directory of the project and add your API keys.
```bash
  GROQ_API_KEY=your_groq_api_key
  GOOGLE_API_KEY=your_google_api_key

```
* Place Your Documents

Ensure your PDF documents are placed in the directory specified in the code. Update the path in the vector_embedding() function if your documents are stored in a different location.


* Prepare the CSS File

Create a style.css file in the root directory to style your Streamlit application. The contents can be customized as needed.

## Running the Application

* To start the Streamlit application, run the following command:

```bash
  streamlit run your_script.py
```
Replace "your_script.py" with the name of your main script file.

## Usage

- Load the Documents:

Click the "Click to load the Documents" button to initialize the vector store with your documents.

- Clear Conversation:
Click the "Clear Conversation" button to reset the chat history.

- Ask Questions:

Type your question in the input box and press Enter.

## File Structure

```bash
  .
├── .env
├── conversation_log.txt
├── README.md
├── requirements.txt
├── style.css
├── your_script.py
└── your_document_directory/
    ├── document1.pdf
    ├── document2.pdf
    └── ...

```

- **.env:** Contains environment variables including API keys.

- **conversation_log.txt:** Logs of the conversation for maintaining context.

- **README.md:** Documentation for the project.

- **requirements.txt:** Lists all the dependencies required for the project.

- **style.css:** CSS file for styling the Streamlit application.

- **your_script.py:** Main script containing the code for the chatbot.

- **your_document_directory/:** Directory where the PDF documents are stored.

## Acknowledgments

 - [Google Generative AI](https://ai.google/discover/generativeai/)
 - [Gemma](https://blog.google/technology/developers/gemma-open-models/)
 - [OpenAI](https://openai.com/)
 - [Langchain](https://www.langchain.com/)
 - [Streamlit](https://streamlit.io/)


## Screenshots

![App Screenshot](https://github.com/SaiTeja250802/Computer-Market-Hub/blob/main/img0.png)
![App Screenshot](https://github.com/SaiTeja250802/Computer-Market-Hub/blob/main/img1.png)
![App Screenshot](https://github.com/SaiTeja250802/Computer-Market-Hub/blob/main/img2.png)
![App Screenshot](https://github.com/SaiTeja250802/Computer-Market-Hub/blob/main/img3.png)


## Features

- Intelligent Document Query
- High Accuracy
- Low Latency
- User-Friendly UI

