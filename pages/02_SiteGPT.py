import os
import streamlit as st

from typing import List, Dict, Any
from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain_core import retrievers

from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory


if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 메시지 리스트 초기화


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs: Dict[str, Any]) -> Dict:
    docs: List[Document] = inputs["docs"]
    question: str = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs: Dict[str, Any]) -> Any:
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup: BeautifulSoup) -> str:
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url: str) -> retrievers:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        web_path=url,
        parsing_function=parse_page,
        filter_urls=[r"^(.*\/(ai-gateway|vectorize|workers-ai)\/).*"],
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store: FAISS = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


# OpenAI API 키 입력 및 설정
with st.sidebar:

    openai_api_key = st.text_input(
        "OpenAI API Key", key="openai_api_key", type="password"
    )
    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.markdown(
        "[GitHub Repository](https://github.com/Luceta/FullStack-gpt/commit/bcaa7afd4007ce34817d7809b829b2ba8ccc48b7)"
    )
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


def save_memory(question, answer):
    existing_data = memory.load_memory_variables({"input": question})
    if "output" in existing_data and existing_data["output"] == answer:
        print("이미 동일한 입력과 출력이 메모리에 저장되어 있습니다.")
        return False  # 저장하지 않음
    else:
        memory.save_context({"input": question}, {"output": answer})


def get_history():
    return memory.load_memory_variables({})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def save_message(message, role, question=False):
    st.session_state["messages"].append({"message": message, "role": role})
    if question:
        save_memory(message, question)


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:

        llm = ChatOpenAI(
            temperature=0.1,
        )

        memory = ConversationSummaryBufferMemory(llm=llm)

        retriever = load_website(url)

        send_message("i'm ready! ask away!", "ai", save=False)
        paint_history()

        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")

            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            ai_answer = result.content.replace("$", "\$")

            # Save and display AI's answer
            save_message(ai_answer, "ai", question=query)
            send_message(ai_answer, "ai")

else:
    st.session_state["message"] = []
