import os
import streamlit as st
from io import BytesIO
from chromadb.config import Settings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch


CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+paraquet',
    persist_directory='db',
    anonymized_telemetry=False
)
check_point = 'MBZUAI/LaMini-T5-738M'
tokenizer = AutoTokenizer.from_pretrained(check_point)
model = AutoModelForSeq2SeqLM.from_pretrained(
    check_point,
    device_map='auto'
)


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2ext-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=512,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def qa_llm(pdf_file):
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load the PDF file using PyPDFLoader
    doc_loader = PyPDFLoader(pdf_file)
    # Split the text into sentences using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter()
    sentences = text_splitter.split_text(doc_loader.load())
    # Add the sentences to the Chroma vector store
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    db.add(sentences)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def main():
    st.title("PDF Question Answering")
    st.write("Upload a PDF file and ask a question about its contents.")
    
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Read the contents of the uploaded file into a BytesIO object
        pdf_bytes = BytesIO(uploaded_file.read())
        
        # Call the qa_llm function with the BytesIO object as the argument
        qa_result = qa_llm(pdf_bytes)
        
        # Display the question and answer results
        for result in qa_result:
            st.write(f"Question: {result['question']}")
            st.write(f"Answer: {result['answer']}")
            st.write("-----")


if __name__ == "__main__":
    main()
