import time
import streamlit as st
import openai as client
import os
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
import json


def send_chat_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def get_ddgSearch(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    keyword = inputs["keyword"]
    return ddg.run(f"research of {keyword}")


def get_wikiSearch(inputs):
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


functions_map = {
    "get_ddgSearch": get_ddgSearch,
    "get_wikiSearch": get_wikiSearch,
    "save_to_file": save_to_file,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ddgSearch",
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
            "name": "get_wikiSearch",
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
    {
        "type": "function",
        "function": {
            "name": "save_to_file",
            "description": "Save information to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query used to retrieve the information.",
                    },
                    "result": {
                        "type": "string",
                        "description": "The result to be saved in the file.",
                    },
                },
                "required": ["query", "result"],
            },
        },
    },
]


def create_thread(content):
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ]
    )
    return thread


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id).data
    messages.reverse()
    for message in messages:
        with st.chat_message(message.role):
            if message.content:
                st.markdown(message.content[0].text.value)
                if hasattr(message.content[0], "text"):
                    if (
                        hasattr(message.content[0].text, "annotations")
                        and message.content[0].text.annotations
                    ):
                        for annotation in message.content[0].text.annotations:
                            st.markdown(f"Source: {annotation.ed}")


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )


def main():
    st.set_page_config(
        page_title="OpenAI Assistant GPT",
        page_icon="🕵️‍♂️",
    )

    st.sidebar.text("OpenAI Assistant")
    st.markdown(
        """
    # OpenAi Assistant GPT

    Welcome to OpenAi Assistant GPT.

    Enter keywords for the research results you want to find.
    """
    )

    api_key = st.sidebar.text_input("Put your OpenAI API Key here", type="password")
    st.sidebar.markdown(
        "[GitHub Repository](https://github.com/Luceta/FullStack-gpt/tree/feat/OpenAI-Assistants)"
    )

    assistant_id = st.session_state.get("assistant_id", "")

    thread_id = st.session_state.get("thread_id", "")
    run_id = st.session_state.get("run_id", "")

    if api_key == "":
        st.warning("Please enter your OpenAI API Key first!!")
    else:
        open_ai_client = client.Client(api_key=api_key)

        if assistant_id == "":
            assistant = open_ai_client.beta.assistants.create(
                name="Research AI Assistant",
                instructions="""You are a helful search manager assistance.
            
            You should try to search information WikiPedia or DuckDuckGo,
            
            If there is a list of website urls in the search results list, please extract the content of each website to text and organize it, 
            This will be documented, so please keep the results organized""",
                model="gpt-3.5-turbo-1106",
                tools=functions,
            )
            st.session_state["assistant_id"] = assistant.id
            st.session_state["assistant_name"] = assistant.name
            assistant_id = assistant.id
            st.write(f"Created a new assistant! ID: {assistant_id}")

    if run_id != "":
        run = get_run(run_id, thread_id)
        if run.status == "completed":
            st.success("The search is completed. The details are as follows")
            get_messages(thread_id)

        elif run.status == "in_progress":
            with st.status("In progress..."):
                st.write("Waiting for the AI to respond...")
                time.sleep(3)
                st.rerun()
        elif run.status == "requires_action":
            with st.status("Processing action..."):
                submit_tool_outputs(run_id, thread_id)
                time.sleep(3)
                st.rerun()

    keyword = st.text_input(
        "Enter a research subject.",
        value=st.session_state.get("input", ""),
        key="input",
    )

    if keyword:
        if thread_id == "":
            thread = open_ai_client.beta.threads.create(
                messages=[{"role": "user", "content": keyword}]
            )
            thread_id = thread.id
            st.session_state["thread_id"] = thread_id
        else:
            send_message(thread_id, keyword)

        run = open_ai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        run_id = run.id
        st.session_state["run_id"] = run_id
        st.session_state["clear"] = True
        st.session_state["keyword"] = keyword
        st.rerun()


if __name__ == "__main__":
    main()
