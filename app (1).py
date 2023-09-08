
import os
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

def main():
    # Ask user for PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf_file is not None:
        # Load PDF file using Langchain
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        embeddings = SentenceTransformerEmbeddings()

        # Create Chroma database from documents
        db = Chroma.from_documents(docs, embeddings, persist_directory='.')
        db.persist()

        # Load T5 model and tokenizer
        checkpoint = 'MBZUAI/LaMini-T5-738M'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

        # Create text generation pipeline
        pipe = pipeline(
            'text2text-generation',
            model=base_model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=True,
            top_p=0.95,
            temperature=0.3
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm, chain_type='stuff',
            retriever=retriever
        )

        # Ask user for question and generate answer
        question = st.text_input("Enter your question")

        if st.button("Generate Answer"):
            generated_text = qa(question)
            answer = generated_text['result']
            st.write(answer)


if __name__ == "__main__":
    main()
