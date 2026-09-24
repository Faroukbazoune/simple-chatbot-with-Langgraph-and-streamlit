import streamlit as st
from learning_langgraph.agentic_chatbot_backend import chatbot, conn, gettig_all_threads
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import uuid
from pprint import pprint

db = conn.cursor()


def generate_id():
    return str(uuid.uuid4())


def add_thread_id(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def new_chat():
    st.session_state["thread_id"] = generate_id()
    st.session_state["message_history"] = []
    add_thread_id(st.session_state["thread_id"])


def load_old_messags(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# these are for the first time run ever
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = gettig_all_threads()
st.title("Chat Bot")
st.write(f"current chat :{st.session_state['thread_id']}")

# add the current thread into the threads list
add_thread_id(st.session_state["thread_id"])

# LOADING THE OLD MESSAGES
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

if st.sidebar.button("New Chat"):
    add_thread_id(st.session_state["thread_id"])
    new_chat()
    st.rerun()


st.sidebar.title("My Conversations")


for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(thread_id, key=thread_id):
        st.session_state["thread_id"] = thread_id

        messages = load_old_messags(thread_id)

        temp_messags = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
                temp_messags.append({"role": role, "content": message.content})
            elif isinstance(message, AIMessage):
                role = "assistant"
                if message.content == []:
                    pass
                else:
                    content = message.content

                    if isinstance(content, str):
                        text = content

                    elif isinstance(content, list):
                        texts = []

                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                texts.append(item.get("text", ""))

                        text = "".join(texts)

                    else:
                        text = str(content)

                    if text:
                        temp_messags.append({"role": "assistant", "content": text})

        st.session_state["message_history"] = temp_messags
        st.rerun()


user_input = st.chat_input("type here ")


if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }
    with st.chat_message("ai"):
        status_holder = {"box": None}

        def ai_only():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if not status_holder["box"]:
                        status_holder["box"] = st.status(
                            f"using {tool_name}", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"using {tool_name}", expanded=True, state="running "
                        )
                elif isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        texts = []
                        for item in content:
                            if isinstance(item, dict) and item["type"] == "text":
                                texts.append(item["text"])

                        yield "".join(texts)

        ai_msg = st.write_stream(ai_only())
        if status_holder["box"]:
            status_holder["box"].update(
                label="Finished using the tool", state="complete", expanded=False
            )
    st.session_state["message_history"].append({"role": "assistant", "content": ai_msg})

state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
pprint(state.values)
