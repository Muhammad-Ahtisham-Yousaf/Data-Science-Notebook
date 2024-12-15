import docx.document
import streamlit as st 
import os
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from streamlit_chat import message




my_openai_key = st.secrets['OPENAI_API_KEY']

def main():
    load_dotenv()
    st.set_page_config("Sylabus GPT")
    st.title("Sylabus GPT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload Your Files here:",type = ['pdf','docx'],accept_multiple_files  = True)
        openai_api_key = my_openai_key
        process = st.button("Run")
        
        
        if process:
            if not uploaded_files:
                st.warning("Please Upload File First,Thanks!")
                st.stop()
            if not openai_api_key:
                st.info("Please provide API key first!")
                st.stop()
    files_text = get_files_text(uploaded_files)
    st.write("Files loaded...")
    # st.write(files_text)
        # Display details of all files
    # st.write("### Uploaded Files Details:")
    # for file_info in files_text:
    #     # st.write(f"Full Name: {file_info}")
    #     # st.write(f"File Name: {file_info[0]}")
        # st.write(f"Extension: {file_info}")
        # st.write("---")  # Separator between files
    text_chunks = get_text_chunks(files_text)
    st.write("Chunks Created...")
    vectorstore = get_vector_store(text_chunks)
    st.write("Vector Store Created...")
    st.session_state.conversation = get_conversation_chain(vectorstore,my_openai_key)
    st.session_state.processComplete = True

    if st.session_state.processComplete == True:
      user_question = st.chat_Input("Enter Prompt")
    if user_question:
        handel_user_input(user_question)


def get_files_text(uploaded_files):
    text = ""
    file_details = []
    for uploaded_file in uploaded_files:
        splited_name = os.path.splitext(uploaded_file.name)
        file_name = splited_name[0]
        file_extension = splited_name[1]
    #     file_details.append((file_name,file_extension))
       
    # return file_extension
        if file_extension == ".pdf":
           text += get_pdf_text(uploaded_file)
           
        if file_extension == ".docx":
            text += get_docx_text(uploaded_file)
    return text

def get_pdf_text(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(uploaded_file):
    doc  = docx.Document(uploaded_file)
    allText = []
    for docparagraph in doc.paragraphs:
        allText.append(docparagraph.text)
    text = " ".join(allText)
    return text


def get_text_chunks(files_text):
    text_splitter = CharacterTextSplitter(separator = "\n",chunk_size = 900)
    text_chunks = text_splitter.split_text(files_text)
    return text_chunks

def get_vector_store(text_chunks):
    text_embedding_model = HuggingFaceEmbeddings(model_name = "aspire/acge_text_embedding" )
    vector_database = FAISS.from_text(text_chunks,text_embedding_model)
    return vector_database


def get_conversation_chain(vectorstore,my_openai_key): #to contact with vector store
    llmodel = ChatOpenAI(openai_api_key = my_openai_key,model = "gpt-3.5-turbo",temperature = 0.5)
    storage = ConversationBufferMemory(memory_key="chat_history",return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llmodel,retriever=vectorstore.as_retriever(),memory = storage)
    return conversation_chain

def handel_user_input(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'Question':user_question})
    st.session_state.chat_history = response.history

    #layout for input/output response container
    response_container = st.container()

    with response_container:
        for i , messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
              message(messages.content,key=str(i),is_user=True)
              message(messages.content,key=str(i))






if __name__ == "__main__":
    main()