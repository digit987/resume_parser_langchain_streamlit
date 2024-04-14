import os
import io
import pandas as pd
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

# Load environment variables
os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"]

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    retrieved_data = FAISS.from_texts(chunks, embeddings)
    
    return retrieved_data

def main():
    st.title("Getting details from resume")
    
    pdfs = st.file_uploader('Upload your PDF Document', type='pdf', accept_multiple_files=True)
    resume_data = [['Contact Number', 'Email Address', 'Text']]
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        retrieved_data = process_text(text)
        
        queries = [r'Give me contact number from this resume. Only write the number, dont't describe it. For example, don't write Contact No.: 123, only 123',
                r'Give me email address from this resume. Only write the email address, dont't describe it. For example, don't write Email Address: abc, only abc',
                r'Give me all the text from this resume']
        
        response_list = []

        for query in queries:
            docs = retrieved_data.similarity_search(query)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type='stuff')

            response = chain.run(input_documents=docs, question=query)
            response_list.append(response)
            #st.write(response)
        resume_data.append(response_list)
    df = pd.DataFrame(resume_data)
    print(resume_data, df)
    df.to_csv("Resume Details.csv", index=False, header=False)
    df = pd.read_csv("Resume Details.csv")
    csv_data = df.to_csv(index=False, header=False).encode()
    if csv_data:
        st.download_button(
            label="Download resume data as CSV",
            data=csv_data,
            file_name='Resume Details.csv',
            mime='text/csv',
            key="download-resume-csv",
        )
            
if __name__ == "__main__":
    main()
