
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



embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:8b")
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


db_connection = get_db_connection()

if db_connection:
    print("Connected to DB")
else:
    print("Failed to connect to DB")



pdf_path = "doubleratchet.pdf"

def load_pdf(file_path):
    return PDFPlumberLoader(file_path).load()


def get_file_name (document):
    if not document or not document[0].page_content:
        return None
    first_page_text = document[0].page_content

    words = re.findall(r'\b(\w+)\b', first_page_text, re.UNICODE)
    if words and len(words) >= 4:
        return ' '.join(words[:4])
    else:
        return None


def split_text(input_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True
    )
    return text_splitter.split_documents(input_text)

def answer_question(question, documents):
    context = "\n\n".join(documents)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})



#delete_file("Garbuz A 2023 Old", db_connection)

# name = fetch_name(db_connection)
# for n in name:
#     print(n[0])

# pdf = load_pdf(pdf_path)
# filename = get_file_name(pdf)
# print(filename)
# chunked_documents = split_text(pdf)
#
#
# store_vectors_in_db(db_connection,embeddings, chunked_documents,filename)
# question = "what is KDF chains in double ratchet algorithm?"
# related_documents = retrieve_docs_from_db(db_connection, embeddings, question)
# print(related_documents)
# print("Answer")
# answer = answer_question(question,related_documents)
# print(answer)





