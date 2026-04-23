import streamlit as st
import os
from pathlib import Path
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS


# ======================
# 캐싱 (속도 핵심)
# ======================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource
def load_llm(api_key):
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )


# ======================
# 기본 문서 로딩
# ======================
def load_default_docs():
    docs_path = "docs"
    doc_list = []

    if not os.path.exists(docs_path):
        return doc_list

    for file_name in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file_name)

        try:
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file_name.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                continue

            doc_list.extend(loader.load_and_split())
        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}")

    return doc_list


# ======================
# chunk
# ======================
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x))
    )
    return splitter.split_documents(text)


# ======================
# vector DB
# ======================
def get_vectorstore(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)


# ======================
# chain 생성
# ======================
def get_chain(vectorstore, llm):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="mmr"),
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        ),
        return_source_documents=True,
        verbose=False
    )


# ======================
# 파일 업로드 처리
# ======================
def load_uploaded_files(files):
    docs = []

    for file in files:
        file_path = file.name

        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            continue

        docs.extend(loader.load_and_split())

    return docs


# ======================
# UI
# ======================
def main():
    st.set_page_config(page_title="MESIM Assistant", page_icon="🤖")
    st.title("MESIM AI Assistant 🤖")

    api_key = st.secrets["google_api_key"]

    embeddings = load_embeddings()
    llm = load_llm(api_key)

    # session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "질문을 입력해주세요!"}
        ]

    if "vectorstore" not in st.session_state:
        default_docs = load_default_docs()
        chunks = get_text_chunks(default_docs)
        st.session_state.vectorstore = get_vectorstore(chunks, embeddings)
        st.session_state.chain = get_chain(st.session_state.vectorstore, llm)

    # sidebar
    with st.sidebar:
        files = st.file_uploader(
            "파일 업로드",
            type=["pdf", "docx", "pptx"],
            accept_multiple_files=True
        )
        if st.button("적용"):
            if files:
                uploaded_docs = load_uploaded_files(files)
                chunks = get_text_chunks(uploaded_docs)

                upload_vs = get_vectorstore(chunks, embeddings)
                st.session_state.vectorstore.merge_from(upload_vs)

                st.session_state.chain = get_chain(
                    st.session_state.vectorstore, llm
                )

                st.success("업로드 완료")

    # chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    query = st.chat_input("질문 입력")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.chain

            with st.spinner("생각 중..."):
                result = chain({"question": query})

                answer = result["answer"]
                docs = result.get("source_documents", [])

                st.markdown(answer)

                if docs:
                    with st.expander("참고 문서"):
                        for d in docs:
                            st.markdown(d.metadata.get("source", "unknown"))


        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()
