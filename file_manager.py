import os
from dotenv import load_dotenv

import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import chromadb

load_dotenv()

def load_chunk_persist_pdf():
    pdf_folder_path = 'C:/Users/tschebesta/Desktop/Chroma_agent/pdfs'
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)

    # Create a new Chroma instance with the appropriate settings
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory="C:/Users/tschebesta/Desktop/app3db"
    )
    vectordb.persist()

    return vectordb

def create_agent_chain():
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query):
    vectordb = load_chunk_persist_pdf()
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer

def get_sources(answer):
    for source in answer['source_documents']:
        source.metadata['source']
    return source


st.set_page_config(page_title="Doc Searcher", page_icon=":milky_way:")
st.header("Deloitte Private Doc Searcher :milky_way:")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(get_sources(get_llm_response(form_input)))
