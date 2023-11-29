import os
import tempfile
import streamlit as st
import tiktoken
from loguru import logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

import sys

def main():
    st.set_page_config(
    page_title="WinC Chat",
    page_icon=":speech_balloon:")
    
    st.title(":speech_balloon: _Wise _In _compaby chatbot/[:blue[Beta]/]")
                    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    with st.sidebar:
        upladed_file = st.file_uploader("Upload your file",type=['pdf','docx','pptx'])
        openai_api_key = st.text_input("OpenAI API key",key='chatbot_api_key',type='password')
    
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role":"product recommender",
                                        "Content":"안녕하세요! 추천받을 상품 정보를 입력해주세요"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    history = StreamlitChatMessageHistory(key="chat_messages")
    
    if query := st.chat_input("질문을 입력해주세요."):
            st.session_state.messages.append({"role":"user", "content":query})
            
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("product recommender"):
                chain = st.session_state.conversation
                
                with st.spinner("Thinking..."):
                    result = chain({"question":query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    
                    st.markdown(response)
            st.session_state.messages.append({"role":"product recommender", "content":response})
 
if __name__ == '__main__':
    main()
