import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from db_connection import get_db_connection
from db_store import store_vectors_in_db;
from db_retrieve import retrieve_docs_from_db
import psycopg2
from db_fetch_file  import fetch_name
from db_delete import delete_file
import re
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in detail. Dont print out <>
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'data/'
txt_directory = 'data/'
db_connection = get_db_connection()
file_names_list = []

if db_connection:
    print("Connected to DB")
else:
    print("Failed to connect to DB")

embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:8b")

st.title(':red[PDF] RAG')
st.write('This is an app to answer questions using data from technical documents, such as user manuals.')
st.write(
    'It is based on the principle of Retrieval-Augmented Generation (:red[RAG]) - a technique that modifies interactions with a language model so that it responds to user queries with reference to a specified set of documents.')
st.write('Upload a file below and try it out for yourself!')
st.divider()
st.write(':gray[Model currently in use:] ' + model.model)



# Custom CSS
st.markdown("""
<style>  
.stButton > button {
    background-color: #6B6D70
    color: white; /* White text */
    border: none; /* Remove borders */
    padding: 5px 5x; 
    width: 100%; /* Full width */
    text-align: center; /* Centered text */
    font-size: 12px; /* Font size */
    margin: 2px; /* Margin */
    cursor: pointer; /* Pointer cursor on hover */
    border-radius: 12px; /* Rounded corners */
}
.stButton > button:hover {
    background-color: #E5E9F0 ; /* Darker green on hover */
}
</style>
""", unsafe_allow_html=True)

def extract_file_name (document):
    if not document or not document[0].page_content:
        return None
    first_page_text = document[0].page_content

    words = re.findall(r'\b(\w+)\b', first_page_text, re.UNICODE)
    if words and len(words) >= 4:
        return ' '.join(words[:4])
    else:
        return None



def upload_pdf(file):
    try:
        with open(pdfs_directory + file.name, "wb") as f:
            f.write(file.getbuffer())
    except Exception as e:
        st.error(f"Error uploading file: {e}")

def upload_txt(file):
    try:
        with open(txt_directory + file.name, "wb") as f:
            f.write(file.getbuffer())
    except Exception as e:
        st.error(f"Error uploading text file: {e}")

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def answer_question(question, documents):
    if isinstance(documents, list) and all(isinstance(doc, str) for doc in documents):
        context = "\n\n".join(documents)
    else:
        context = "\n\n".join([doc.page_content for doc in documents])

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})

    response_cleaned = response.replace("<think>", "").replace("</think>", "")

    return response_cleaned

def read_txt_to_array(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines

def save_answers_to_txt(filename, answers):
    with open(filename, "w", encoding="utf-8") as file:
        for answer in answers:
            file.write(answer + "\n")

def render_doc_names(names):
    cleaned_names = [name[0] for name in names]

    st.sidebar.header("Uploaded files")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    for file_name in cleaned_names:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(file_name)

        if col2.button("Delete", key=file_name):
            delete_file(file_name, db_connection)
            cleaned_names.remove(file_name)

on = st.toggle("Upload question file")

uploaded_file = st.file_uploader("Upload PDF", type="pdf",accept_multiple_files=False)




if uploaded_file:
    upload_pdf(uploaded_file)
    print(f"Uploaded file path: {pdfs_directory + uploaded_file.name}")  # Debugging line
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    file_name = extract_file_name(documents)
    print(file_name)
    chunked_documents = split_text(documents)
    store_vectors_in_db(db_connection, embeddings, chunked_documents, file_name)
    st.chat_message("user").write(f"Finished indexing {uploaded_file.name}")
    st.chat_message("assistant").write("Ready to answer questions")

    st.session_state.file_uploaded = True

if 'file_uploaded' in st.session_state and st.session_state.file_uploaded:
    file_names_list.append(file_name)
    st.session_state.file_uploaded = False




if on:
    question_file = st.file_uploader(
        "Upload questions TXT",
        type="txt",
        accept_multiple_files=False
    )
    answers = []
    if question_file:
        upload_txt(question_file)
        lines_array = read_txt_to_array(txt_directory + question_file.name)
        for question in lines_array:
            st.chat_message("user").write(f"Asking question:  {question}")
            related_documents = retrieve_docs_from_db(db_connection,embeddings, question)
            answer = answer_question(question, related_documents)
            answers.append(answer)
            st.chat_message("assistant").write(answer)

        output_filename = "answers.txt"
        if output_filename:
            save_answers_to_txt(output_filename, answers)


file_names_list = fetch_name(db_connection)
render_doc_names(file_names_list)
question = st.chat_input()
if question is not None:
    st.chat_message("user").write(f"Asking question: {question}")
    related_documents = retrieve_docs_from_db(db_connection, embeddings, question)
    print(related_documents)
    answer = answer_question(question, related_documents)
    print(question)
    answer = answer_question(question, related_documents)
    print(answer)
    st.chat_message("assistant").write(answer)
