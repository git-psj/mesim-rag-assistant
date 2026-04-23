import streamlit as st
import tiktoken
from loguru import logger

import os
from pathlib import Path

from langchain.chains import ConversationalRetrievalChain
#from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
#from langchain.callbacks import get_openai_callback
#from langchain.memory import StreamlitChatMessageHistory

# 메인 함수
def main():
  if "conversation" not in st.session_state:
      st.session_state.conversation = None
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = None
  #if "processComplete" not in st.session_state:
  #    st.session_state.processComplete = None
    
  st.set_page_config(
    page_title="MESIM Assistant",
    page_icon="🤖")
  st.title("_MESIM 운영 지원 :red[AI Assistant]_🤖")

  google_api_key = st.secrets['google_api_key']

  default_docs = load_default_docs()
  embeddings = HuggingFaceEmbeddings(
      model_name="jhgan/ko-sroberta-multitask",
      model_kwargs={'device': 'cpu'},
      encode_kwargs={'normalize_embeddings': True}
  )
  text_chunks = get_text_chunks(default_docs)
  should_load = False
  
  if os.path.exists("faiss_index"):
    docs_time = get_latest_docs_modified_time()
    faiss_time = get_faiss_modified_time()

    # docs 변경 없으면 기존 FAISS 사용
    if docs_time < faiss_time:
        should_load = True

  if should_load:
      vectorstore = FAISS.load_local(
          "faiss_index",
          embeddings,
          allow_dangerous_deserialization=True
      )
  else:
      vectorstore = get_vectorstore(text_chunks, embeddings)
      vectorstore.save_local("faiss_index")

  if st.session_state.conversation is None:
    st.session_state.conversation = get_conversation_chain(vectorstore,google_api_key) 

  

  with st.sidebar:
      uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
      process = st.button("Process")
  if process:
    if not uploaded_files:
      st.warning("업로드된 파일이 없습니다.")
      st.stop()
    files_text = get_text(uploaded_files) if uploaded_files else []
    if files_text:
      text_chunks = get_text_chunks(files_text)
      upload_vectorstore = get_vectorstore(text_chunks, embeddings)
      # 기본 문서 + 업로드 문서 병합
      vectorstore.merge_from(upload_vectorstore)
    #st.session_state.processComplete = True
    st.session_state.conversation = get_conversation_chain(vectorstore,google_api_key) 
  
  

  if 'messages' not in st.session_state:
      st.session_state['messages'] = [{"role": "assistant", 
                                      "content": "안녕하세요! MESIM Assistant입니다. 질문 사항을 적어주세요!"}]

  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])

  #history = StreamlitChatMessageHistory(key="chat_messages")

# Chat logic
if query := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        chain = st.session_state.conversation

        with st.spinner("Thinking..."):
            try:
                result = chain({"question": query})
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                return

            st.session_state.chat_history = result.get('chat_history', [])

            response = result.get('answer', 'No answer provided')
            source_documents = result.get('source_documents', [])

            st.markdown(response)
            if source_documents:
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata.get('source', 'No source metadata'), help=doc.page_content)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks, embeddings):
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore,google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model='gemini-1.5-flash')
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(search_type='mmr', verbose=True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Failed to create conversation chain: {str(e)}")
        return None

# 기본 문서 로딩
def load_default_docs():
    docs_path = "docs"
    doc_list = []
  
    if not os.path.exists(docs_path):
        return doc_list
      
    for file_name in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file_name)

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents = loader.load_and_split()
        elif file_name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_path)
            documents = loader.load_and_split()
        else:
            continue
        doc_list.extend(documents)

    return doc_list

# 백터 DB 생성시간 확인
def get_faiss_modified_time():
    faiss_path = Path("faiss_index")
    if not faiss_path.exists():
        return 0
    latest_time = 0
    for file in faiss_path.iterdir():
        if file.is_file():
            latest_time = max(
                latest_time,
                file.stat().st_mtime
            )
    return latest_time

def get_latest_docs_modified_time():
    docs_path = Path("docs")
    if not docs_path.exists():
        return 0
    latest_time = 0
    for file in docs_path.iterdir():
        if file.is_file():
            latest_time = max(
                latest_time,
                file.stat().st_mtime
            )
    return latest_time
  
if __name__ == '__main__':
    main()
