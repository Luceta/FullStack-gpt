import streamlit as st
import json
import os
import openai as client
import time

from langchain.utilities import WikipediaAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


from langchain_community.agent_toolkits import FileManagementToolkit
from selenium import webdriver

from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from bs4 import BeautifulSoup


# ============================================================
# 커스텀 도구 정의


def get_web_content_with_scroll(inputs):
    url = inputs["url"]

    # Selenium WebDriver 설정
    options = webdriver.ChromeOptions()
    options.headless = True  # 창을 표시하지 않음
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    try:
        driver.get(url)
        sleep(2)  # 페이지가 로딩될 때까지 잠시 대기

        # 페이지 끝까지 스크롤
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            # 스크롤을 아래로 내리기
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(2)  # 콘텐츠가 로딩될 때까지 대기

            # 새로운 스크롤 높이를 가져와 이전 높이와 비교
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  # 더 이상 스크롤할 수 없으면 종료
            last_height = new_height

        # 페이지에서 텍스트 콘텐츠 가져오기
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for header in soup.find_all(["header", "footer", "nav"]):
            header.decompose()  # 불필요한 요소 제거

        content = soup.get_text(separator="\n", strip=True)
        return content

    except Exception as e:
        print(f"ERROR on get_web_content_with_scroll: {e}")
        return f"Error getting content from {url}. Please try another URL."
    finally:
        driver.quit()


def send_chat_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def get_ddg_search(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    keyword = inputs["keyword"]
    return ddg.run(f"research of {keyword}")


def get_wiki_search(inputs):
    wiki = WikipediaAPIWrapper()
    keyword = inputs["keyword"]
    return wiki.run(f"research of {keyword}")


def save_to_file(query, result):
    working_directory = os.getcwd()
    tools = FileManagementToolkit(
        root_dir=working_directory,
        selected_tools=["write_file"],
    ).get_tools()

    (write_tool,) = tools

    results = write_tool.invoke({"file_path": f"{query}.txt", "text": result["output"]})
    return results


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        st.write(f"Calling function: {function.name} with arg {function.arguments}")
        print(f"Calling function: {function.name} with arg {function.arguments}")
        output = functions_map[function.name](json.loads(function.arguments))

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"output:{output}")
        outputs.append(
            {
                "output": output,
                "tool_call_id": action_id,
            }
        )
    return outputs


def get_assistant():
    if "assistant" in st.session_state:
        return st.session_state["assistant"]

    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_ddg_search",
                "description": "find the duckduck go search Result in this website with keyword arugment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "The research of keyword",
                        }
                    },
                    "required": ["keyword"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_wiki_search",
                "description": "find the wikipedia search Result in this website with query arugment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "The research of keyword",
                        }
                    },
                    "required": ["keyword"],
                },
            },
        },
    ]

    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions="""
        You are a research expert.

        Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 

        When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 

        Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.

        Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.

        The information from Wikipedia must be included.
        """,
        model="gpt-4o-mini",
        tools=functions,
    )

    st.session_state["assistant"] = assistant

    return st.session_state["assistant"]


def get_thread_id(client):
    if "thread_id" not in st.session_state:
        # 'thread_id'가 없다면, 새로 생성하고 세션에 저장
        thread = client.beta.threads.create(
            messages=[{"role": "assistant", "content": "Hi, How can I help you?"}]
        )
        st.session_state["thread_id"] = thread.id

    # thread_id가 이미 존재하면 그 값을 반환
    return st.session_state["thread_id"]


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )


def get_messages(thread_id):
    messages = list(client.beta.threads.messages.list(thread_id=thread_id))
    return list(reversed(messages))


def start_run(thread_id, assistant_id, content):

    if "run" not in st.session_state:
        # Handle task execution and tool calls
        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        st.session_state["run"] = run

    else:
        # 이미 진행 중인 작업이 있으면 계속 사용
        run = st.session_state["run"]
        print("Continuing with the existing run.")

    with st.status("Processing..."):
        while get_run(run.id, thread_id).status == "requires_action":
            submit_tool_outputs(run.id, thread_id)

    # 상태 확인 후 처리
    print(f"done, {get_run(run.id, thread_id).status}")
    final_message = get_messages(thread_id)[-1]

    if get_run(run.id, thread_id).status == "completed":
        # 'completed' 상태일 때 채팅 메시지로 출력
        with st.chat_message(final_message.role):
            st.markdown(final_message.content[0].text.value)

        print(final_message)

    elif get_run(run.id, thread_id).status == "failed":
        # 'failed' 상태일 때 채팅 메시지로 출력
        with st.chat_message("assistant"):
            st.markdown("Sorry. I failed researching. Try Again later :()")

    return run


def get_thread_id(client):
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create(
            messages=[{"role": "assistant", "content": "Hi, How can I help you?"}]
        )
        print(thread)
        st.session_state["thread_id"] = thread.id

    return st.session_state["thread_id"]


# ============================================================


# functions_map 정의
functions_map = {
    "get_ddg_search": get_ddg_search,
    "get_wiki_search": get_wiki_search,
    "save_to_file": save_to_file,
    # "get_web_content_with_scroll": get_web_content_with_scroll,
}


# ============================================================

st.sidebar.text("OpenAI Assistant")
st.markdown(
    """
# OpenAi Assistant GPT

Welcome to OpenAi Assistant GPT.

Enter keywords for the research results you want to find.
"""
)

with st.sidebar:
    api_key = st.sidebar.text_input("Put your OpenAI API Key here", type="password")
    if not api_key:
        st.error("OpenAI API Key is required.")

    st.markdown("---")
    st.markdown(
        "[GitHub Repository](https://github.com/Luceta/FullStack-gpt/tree/feat/OpenAI-Assistants)"
    )

if api_key:
    # open_ai_client = client.Client(api_key=api_key)

    # API 키로 클라이언트 초기화
    open_ai_client = client.Client(api_key=api_key)

    keyword = st.chat_input("Ask a question to the website.")

    # 대화 기록을 저장하는 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 표시
    # 메시지 기록 출력
    for idx, message in enumerate(get_messages(get_thread_id(open_ai_client))):
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value)

    if keyword:
        assistant = get_assistant()
        thread_id = get_thread_id(client=open_ai_client)
        send_chat_message(keyword, "user")
        start_run(thread_id, assistant.id, keyword)
        # send_chat_message("user")
