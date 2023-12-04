import streamlit as st

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain import OpenAI
from dotenv import load_dotenv

def main() :

    st.set_page_config(
    page_title="WinC Chat",
    page_icon=":speech_balloon:")
    
    st.title(":speech_balloon: Wise InCompany chatbot[:blue[Beta]]")

    openai_key = st.secrets["openai"]["openai_api_key"]
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key"]


    csv_file = st.file_uploader('train.csv', type='csv')

    if csv_file is not None:
        user_quesion = st.text_input("상품 검색어 :")

        llm=OpenAI(temperature=0)
        agent=create_csv_agent(llm,csv_file, verbose=True)

        if user_quesion is not None and user_quesion !="":
            response = agent.run(user_quesion)
            st.write(response)

if __name__ == '__main__':
    main()