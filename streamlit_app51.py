import streamlit as st

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain import OpenAI

def main() :

    st.set_page_config(
    page_title="WinC Chat",
    page_icon=":speech_balloon:")
    
    st.title(":speech_balloon: Wise InCompany chatbot[:blue[Beta]]")

    filePath="data.csv"

    if filePath is not None:
        llm=OpenAI(temperature=0)
        agent=create_csv_agent(llm,
                               filePath,
                               verbose=True,
                               )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 원하시는 상품정보를 알려주세요."}]
    
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            response = agent.run({"question": query})
            st.write(response)
    
if __name__ == '__main__':
    main()
