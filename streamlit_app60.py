import streamlit as st
from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def main() :
    
    st.set_page_config(
    page_title="WinC Chat",
    page_icon=":speech_balloon:")
    
    st.title(":speech_balloon: Wise InCompany chatbot[:blue[Beta]]")
    filePath="data.csv"
    DB_FAISS_PATH = "vectorstor/db_faiss"

    if filePath is not None:
        loader = CSVLoader(file_path=filePath,encoding='utf-8',csv_args={
            'delimiter':','
        } )
        data=loader.load()
        st.json(data)
        embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                         model_kwargs={'device':'cpu'})
        db=FAISS.from_documents(data,embeddings)
        db.save_local(DB_FAISS_PATH)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    
        chain=ConversationalRetrievalChain.from_llm(llm=llm,
                                                    retriever=db.as_retriever())
        
        def conversational_chat(query):
            result=chain({"question":query,"chat_history":st.session_state['history']})
            st.session_state['history'].append((query,result["answer"]))
            return result["answer"]
        
        if 'history' not in st.session_state:
            st.session_stae['history']=[]
        
        if 'generated' not in st.session_state:
            st.session_state['generated']=["원하시는상품정보를 알려주세요"]
        
        if 'past' not in st.session_state:
            st.session_state['past']=["TT!!!!"]

        response_container = st.container()

        container=st.container()

        with container:
            with st.form(key="my_from", clear_on_submit=True):
                user_input = st.text_input("Query:",placeholder="데이터입력",
                                           key='input')
                submit_button=st.form_submit_button(label="chat")
            if submit_button and user_input:
                output=conversational_chat(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user',
                            avatar_style='big-smile')
                    message(st.session_state['past'][i], is_user=True, key=str(i))
            
if __name__ == '__main__':
    main()
