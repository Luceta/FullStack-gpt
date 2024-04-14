import streamlit as st
import openai as client


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def create_chat_ui():

    st.markdown(
        """
    # OpenAi Assistant GPT

    Welcome to OpenAi Assistant GPT.

    Enter keywords for the research results you want to find.
    """
    )

    api_key = st.sidebar.text_input("Put your OpenAI API Key here", type="password")
    st.sidebar.markdown(
        "[GitHub Repository](https://github.com/Luceta/FullStack-gpt/tree/feat/quizGPT)"
    )

    if api_key == "":
        st.warning("Please enter your OpenAI API Key first!!")
    else:
        openai_client = client.Client(api_key=api_key)
        # paint_history(st.session_state.get("messages", []))  # 이전 대화 기록을 표시
        send_message("I'm ready! Ask away!", "ai", save=False)
        message = st.chat_input("Ask anything about your file...")

        if message:
            send_message("I' answer!", "ai", save=False)

            # st.success("The search is in progress. Please wait for the AI to respond.")


def main():
    st.set_page_config(
        page_title="OpenAI Assistant GPT",
        page_icon="🕵️‍♂️",
    )

    st.sidebar.text("OpenAI Assistant")
    create_chat_ui()


if __name__ == "__main__":
    main()
