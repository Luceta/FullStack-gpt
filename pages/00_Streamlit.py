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

st.title("DocumentGPT")

# OpenAI API Key 입력
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# API 키가 없으면 경고 표시
if not api_key:
    st.sidebar.warning("Please enter your OpenAI API Key.")
else:
    openai.api_key = api_key

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # 최신 모델을 사용합니다
            messages=[{"role": "system", "content": "Test"}],
        )

        # API 키가 유효한 경우
        if response:
            st.sidebar.success("API Key is valid!")  # 유효한 API 키일 때 메시지 출력
            llm = ChatOpenAI(temperature=0.1, streaming=True, openai_api_key=api_key)
            # API 키 설정
    except openai.AuthenticationError:
        st.sidebar.error("Invalid API Key! Please check your key and try again.")
        llm = None  # 인증 오류가 나면 llm 초기화하지 않음
        file = None  # 파일 업로드 UI를 비활성화


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
        "[View Code on GitHub](https://github.com/yourrepo)"
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
