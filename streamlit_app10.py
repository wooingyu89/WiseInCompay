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
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import create_pandas_dataframe_agent

def main():
    st.set_page_config(
    page_title="WinC Chat",
    page_icon=":speech_balloon:")
    
    st.title(":speech_balloon: Wise InCompany chatbot[:blue[Beta]]")

    df = pd.read_csv('data.csv')


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
                source_documents = result['source_documents']

                st.markdown(response)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "product recommender", "content": response})



if __name__ == '__main__':
    main()
