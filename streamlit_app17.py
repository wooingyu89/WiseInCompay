import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
import os
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

def main():
    st.set_page_config(
    page_title="WinC Chat",
    page_icon=":speech_balloon:")
    
    st.title(":speech_balloon: Wise InCompany chatbot[:blue[Beta]]")

    os.environ["OPENAI_API_KEY"] = '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'

    #Load Data with LangChain CSVLoader
    loaders=CSVLoader('data.csv', encoding='utf-8')
    docs=loaders.load()
    # Load the Data
    data=docs
    #Split the Text into Chunks
    text_chunks = get_text_chunks(docs)
    #Create a Vector Store
    vectorstore=get_vector_store(text_chunks)
    
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None\
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "product recommender", 
                                        "content": "안녕하세요! 원하시는 상품정보를 알려주세요."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("product recommender"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                
                st.markdown(response)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "product recommender", "content": response})

#Prepare data for embedding
def get_text_chunks(docs):
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks=text_splitter.split_documents(docs)
    return text_chunks

#Embed the data in FAISS
def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(text_chunks, embeddings)
    
    return vectorstore

#Create a Conversation Chain
def get_conversation_chain(vectorstore):
    llm=ChatOpenAI(temperature=0.0)
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

if __name__ == '__main__':
    main()
