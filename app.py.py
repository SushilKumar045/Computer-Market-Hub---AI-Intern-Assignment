import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt_template_str = """
You are an AI assistant with access to a collection of documents. Answer the questions based on the provided context only. Focus on relevant sections, headings, or keywords to provide the most accurate response. Use the following guidelines:

1. Identify and extract the sections or paragraphs that contain the key information related to the question.
2. If a heading or keyword such as "red wine", "best wine", "Pinot Noir", or similar terms are present, prioritize the content under those headings.
3. Provide a concise and informative answer based on the extracted content.
4. Maintain context from previous questions to ensure coherent responses in follow-up questions.

<context>
{context}
<context>

Questions: {input}
"""

def embed_with_retries(embedding_function, texts, retries=3):
    """Retry embedding function in case of errors."""
    for attempt in range(retries):
        try:
            return embedding_function(texts)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)  
            else:
                st.error(f"Embedding failed after {retries} attempts: {e}")
                return None

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("your_path_to_the_pdf_document")  # Replace with actual path of your document 
        st.session_state.docs = st.session_state.loader.load() 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  
        
        embeddings = embed_with_retries(st.session_state.embeddings.embed_documents, [doc.page_content for doc in st.session_state.final_documents])
        if embeddings is not None:
            st.session_state.vectors = FAISS.from_texts([doc.page_content for doc in st.session_state.final_documents], st.session_state.embeddings) 

def update_conversation_log(question, answer):
    """Append the question and answer to the log file."""
    with open("conversation_log.txt", "a") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n")
        f.write("-" * 80 + "\n")

def get_context_from_log():
    """Read the log file and return the context."""
    context = ""
    if os.path.exists("conversation_log.txt"):
        with open("conversation_log.txt", "r") as f:
            context = f.read()
    return context

st.markdown('<div class="embedding-section">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("Click to load the Documents"):
        vector_embedding()
        st.write("Vector Store DB Is Ready")
with col2:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.previous_question = ""
        with open("conversation_log.txt", "w") as f:
            f.write("")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('<div class="header"><div class="title">AI Assistant</div><div class="status">Online</div></div>', unsafe_allow_html=True)

st.markdown('<div class="chat-content">', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    st.markdown(f'<div class="message user"><div class="content">{chat["question"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="message bot"><div class="content">{chat["answer"]}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)
prompt1 = st.text_input("Message AI Assistant", key="input_question", on_change=lambda: setattr(st.session_state, 'question_entered', True), placeholder="Enter your question here...")
st.markdown('</div>', unsafe_allow_html=True)

if "question_entered" in st.session_state and st.session_state.question_entered:
    st.session_state.question_entered = False
    
    context = get_context_from_log()
    
    
    if "previous_question" not in st.session_state:
        st.session_state.previous_question = ""

    
    if any(keyword in prompt1.lower() for keyword in ["it", "that", "those"]):
        prompt1 = f"{st.session_state.previous_question}. {prompt1}"

    
    prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
    enhanced_prompt = prompt_template.format_prompt(context=context, input=prompt1)

    document_chain = create_stuff_documents_chain(llm, prompt_template)  # Pass the prompt_template
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    try:
        response = retrieval_chain.invoke({'input': prompt1, 'context': context})
        answer = response['answer']
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        answer = "Sorry, there was an issue retrieving the answer."

    if "The provided text does not contain any information regarding" in answer or "Sorry, I couldn't find specific information" in answer:
        answer = "Please contact our business directly for further assistance."

    st.session_state.chat_history.append({"question": prompt1, "answer": answer})
    st.markdown(f'<div class="message user"><div class="content">{prompt1}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="message bot"><div class="content">{answer}</div></div>', unsafe_allow_html=True)

    update_conversation_log(prompt1, answer)
    st.session_state.previous_question = prompt1 

    end = time.process_time()
    print(end-start)

st.markdown('</div>', unsafe_allow_html=True)
