import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

#Load Nvidia API Key

os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')

llm=ChatNVIDIA(model="nvidia/llama-3.1-nemotron-ultra-253b-v1", ) # this is specifically an llm we are usuing in our project 

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        
st.title("NVIDIA NIM")

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")


prompt1=st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embeddings()
    st.write("FAISS Vector Store DB Is Ready Using NvidiaEmbedding")
    
    
import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm, prompt=prompt)
    retriver=st.session_state.vectors.as_retriever()
    retrival_chain=create_retrieval_chain(retriver, document_chain)
    start=time.process_time()
    response=retrival_chain.invoke({'input':prompt1})
    print("Response time : ", time.process_time().start)
    st.write(response['answer'])
    
    
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("----------------------------")