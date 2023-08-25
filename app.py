import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import whisper
import translators as ts
import pandas as pd
import time
import os

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.emmegi.co.uk/wp-content/uploads/2019/01/User-Icon.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def handle_predefined_question(question):
    time.sleep(30)
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']
    st.sidebar.write(f"Q: {question}")   
    if st.session_state.chat_history and st.session_state.chat_history[-1].type == 'ai':
        st.sidebar.write(f"A: {st.session_state.chat_history[-1].content}")



def main():

    st.set_page_config(page_title="Chat Bot by Data Science",
                       page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "audio_files" not in st.session_state:
        st.session_state.audio_files = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "audio_files" not in st.session_state:
        st.session_state.audio_files = None

    st.header("Chat with multiple Audios 🎝")
    st.subheader("Get answers about your documents:")
 
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        if st.session_state.audio_files and st.session_state.conversation:  # Verifica si se ha cargado un archivo de audio
            handle_userinput(user_question)  # Muestra la respuesta fuera del sidebar
        else:
            st.warning("Please upload/process an audio file first.")  # Muestra una advertencia si no se ha cargado un archivo

    with st.sidebar:
        
        #OPENAI_API_KEY = st.text_input("Introduce tu OPENAI_API_KEY:")
        os.environ["OPENAI_API_KEY"] = st.text_input("Introduce tu OPENAI_API_KEY:")

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.text_input("Introduce tu HUGGINGFACEHUB_API_TOKEN:") 

        st.subheader("Description")
        st.markdown('This app created by the Data Science team implements a chatbot in Streamlit that answers questions about audio documents. It uses Streamlit for the web interface and Whisper for audio transcription. The transcribed audios are split into chunks using LangChain, and embeddings for these chunks are generated using OpenAIEmbeddings. These embeddings are stored in a "vectorstore" using FAISS. The chatbot employs LangChain ChatOpenAI to provide answers based on the embeddings. Additionally, there are rate limits implemented by waiting between predefined questions 20 seconds per query. Driving the rpm of 3/min.')
        st.subheader("Your documents")
        audio_files = st.file_uploader("Upload your Audios here and click on 'Process",type=["wav","mp3","m4a"])

        if audio_files:
            st.session_state.audio_files = audio_files

        if st.button("Process"):
            with st.spinner("Processing"):

                model = whisper.load_model("base")
                st.warning("Upload in progress please wait.")

                start_time = time.time()
                transcription = model.transcribe(audio_files.name, fp16=False, verbose=True)
                elapsed_time = time.time() - start_time

                st.write(f"Transcription took {elapsed_time:.2f} seconds.")
                st.success("File Uploaded Successfully")

                raw_text = transcription['text']
                st.write("This is the transcribed text")
                st.text(raw_text)
                try:
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)

                    st.session_state.audio_processed = True
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                    predefined_questions = ["The Associate informed the call was being recorded?",
                        "The Associate identified themselves with First and Last Name?",
                        "The Associate stated the reason of the call and provided proper branding. (Standard Health, Estrella Insurance)?"
                        
                        ]
                    
                    st.write("These are the questions available in the chat history:")
                    if st.session_state.audio_processed:
                        for question in predefined_questions:
                            st.session_state.chat_history = None
                            handle_predefined_question(question)

                except Exception as e:
                    st.error("Incorrect API key provided.You can find your API key at https://platform.openai.com/account/api-keys.")
                        

if __name__ == '__main__':
    main()
