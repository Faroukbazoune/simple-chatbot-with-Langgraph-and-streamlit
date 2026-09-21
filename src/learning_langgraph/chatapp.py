import streamlit as st
from learning_langgraph.agentic_chatbot_backend import chatbot, conn
from langchain_core.messages import HumanMessage
import uuid

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
    print(state)
    return state.values.get("messages", [])


# these are for the first time run ever


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [
        thread[0]
        for thread in db.execute(
            "SELECT distinct thread_id from checkpoints"
        ).fetchall()
    ]
st.title("Chat Bot")
st.write(f"current chat :{st.session_state['thread_id']}")


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
            else:
                role = "assistant"
            temp_messags.append({"role": role, "content": message.content})

        st.session_state["message_history"] = temp_messags
        st.rerun()


user_input = st.chat_input("type here ")


if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    with st.chat_message("ai"):
        ai_msg = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )
    st.session_state["message_history"].append({"role": "assistant", "content": ai_msg})
