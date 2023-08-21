import os
import streamlit as st
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
    device_map='auto',
    torch_dtype=torch.float32
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


def qa_llm(pdf_path):
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load the PDF file using PyPDFLoader
    doc_loader = PyPDFLoader(pdf_path)
    # Split the text into sentences using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter()
    sentences = text_splitter.split_text(doc_loader.load())
    # Add the sentences to the Chroma vector store
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    db.add(sentences)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def process_answer(instruction, pdf_path):
    response = ''
    instruction = instruction
    qa = qa_llm(pdf_path)
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


def main():
    st.set_page_config(page_title="Search Your PDF 🐦📄", page_icon="🔍")
    st.title("Search Your PDF 🐦📄")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
            """
        )
    pdf_path = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_path is not None:
        question = st.text_area("Enter your Question")
        if st.button("Ask"):
            st.info("Your Question: " + question)
            st.info("Your Answer")
            try:
                answer = process_answer(question, pdf_path)
                st.write(answer)
            except Exception as e:
                st.error(str(e))


if __name__ == "__main__":
  main()
