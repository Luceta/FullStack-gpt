import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import openai
import os


st.title("Streamlit is 🔥 challenage")

# OpenAI API Key 입력
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# 환경 변수에 API 키 설정
os.environ["OPENAI_API_KEY"] = api_key


# API 키가 없으면 경고 표시
if not api_key:
    st.sidebar.warning("Please enter your OpenAI API Key.")
else:
    openai.api_key = api_key
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
    )


# 파일 임베딩 함수
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    return vectorstore.as_retriever()


# 메시지 저장 함수
def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})


# 메시지 출력 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# 채팅 기록 출력
def paint_history():
    for message in st.session_state.get("messages", []):
        send_message(message["message"], message["role"], save=False)


# 문서 형식화 함수
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 챗봇 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up. Context: {context}",
        ),
        ("human", "{question}"),
    ]
)

# 사용자 인터페이스 안내
st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    Upload your files on the sidebar.
"""
)

# 사이드바에서 파일 업로드 및 GitHub 링크
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt, .pdf, or .docx file", type=["pdf", "txt", "docx"]
    )
    st.sidebar.markdown(
        "[View Code on GitHub](https://github.com/Luceta/FullStack-gpt/commit/53a027b19a03ea9cf1fd9f53072a25913abc576c)"
    )  # 실제 GitHub 링크로 교체

# 파일 업로드 및 처리
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")

        # 챗봇 실행
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")
else:
    st.session_state["messages"] = []  # 파일이 없을 때 채팅 기록 초기화
