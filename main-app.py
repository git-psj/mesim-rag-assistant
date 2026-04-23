import streamlit as st
from loguru import logger

import os
from pathlib import Path

from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

def count_tokens(text):
    """tiktoken 대신 단순 문자 기반 토큰 추정 (Gemini는 tiktoken 미지원)"""
    return len(text) // 4  # 평균적으로 4자 = 1토큰 근사치

def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(
        page_title="MESIM Assistant",
        page_icon="🤖")
    st.title("MESIM :red[AI Assistant] 🤖")

    gemini_api_key = st.secrets['gemini_api_key']

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
        
    if vectorstore is not None and gemini_api_key:
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(vectorstore, gemini_api_key)

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")

    if process:
        if not uploaded_files:
            st.warning("업로드된 파일이 없습니다.")
            st.stop()
        files_text = get_text(uploaded_files) if uploaded_files else []
        if files_text:
            text_chunks = get_text_chunks(files_text)
            upload_vectorstore = get_vectorstore(text_chunks, embeddings)
            vectorstore.merge_from(upload_vectorstore)
        st.session_state.conversation = get_conversation_chain(vectorstore, gemini_api_key)

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                          "content": "안녕하세요! Gemini 기반 MESIM AI Assistant입니다. 질문 사항을 적어주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})


def count_tokens(text):
    return len(text) // 4


def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
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
        length_function=count_tokens  # tiktoken → 자체 함수로 교체
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks, embeddings):
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vectorstore, gemini_api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",       # ← 선택한 모델
        google_api_key=gemini_api_key,
        temperature=0,
        convert_system_message_to_human=True  # Gemini는 system role 미지원 → 필수 옵션
    )
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


def get_faiss_modified_time():
    faiss_path = Path("faiss_index")
    if not faiss_path.exists():
        return 0
    latest_time = 0
    for file in faiss_path.iterdir():
        if file.is_file():
            latest_time = max(latest_time, file.stat().st_mtime)
    return latest_time


def get_latest_docs_modified_time():
    docs_path = Path("docs")
    if not docs_path.exists():
        return 0
    latest_time = 0
    for file in docs_path.iterdir():
        if file.is_file():
            latest_time = max(latest_time, file.stat().st_mtime)
    return latest_time


if __name__ == '__main__':
    main()
