import os
import uuid
import streamlit as st

from learning_langgraph.tools import ingesting_into_rag
from learning_langgraph.agentic_chatbot_backend import (
    chatbot,
    conn,
    gettig_all_threads,
)

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    AIMessage,
)

db = conn.cursor()


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def generate_id():
    return str(uuid.uuid4())


def add_thread_id(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def new_chat():
    st.session_state["thread_id"] = generate_id()
    st.session_state["message_history"] = []

    add_thread_id(st.session_state["thread_id"])


def load_old_messages(thread_id):

    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})

    return state.values.get("messages", [])


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = gettig_all_threads()


# Add current thread
add_thread_id(st.session_state["thread_id"])


# =========================================================
# PAGE
# =========================================================

st.title("Chat Bot")

st.write(f"Current chat: {st.session_state['thread_id']}")


# =========================================================
# DISPLAY CURRENT MESSAGES
# =========================================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
        else:
            st.text(message["content"])


# =========================================================
# SIDEBAR
# =========================================================

if st.sidebar.button("New Chat"):

    new_chat()

    st.rerun()


st.sidebar.title("My Conversations")


for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(thread_id, key=f"thread_{thread_id}"):

        st.session_state["thread_id"] = thread_id

        messages = load_old_messages(thread_id)

        temp_messages = []

        for message in messages:

            # ---------------------------------------------
            # USER MESSAGE
            # ---------------------------------------------

            if isinstance(message, HumanMessage):

                temp_messages.append(
                    {
                        "role": "user",
                        "content": message.content,
                    }
                )

            # ---------------------------------------------
            # AI MESSAGE
            # ---------------------------------------------

            elif isinstance(message, AIMessage):

                if message.content == []:
                    continue

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

                    temp_messages.append(
                        {
                            "role": "assistant",
                            "content": text,
                        }
                    )

        st.session_state["message_history"] = temp_messages

        st.rerun()


# =========================================================
# CHAT INPUT + FILE UPLOAD
# =========================================================

chat_input = st.chat_input(
    "Type here...",
    accept_file=True,
    file_type=["pdf"],
)


# =========================================================
# HANDLE CHAT INPUT
# =========================================================

if chat_input:

    user_input = chat_input.text

    uploaded_files = chat_input.files

    # =====================================================
    # HANDLE FILE UPLOAD
    # =====================================================

    if uploaded_files:

        for uploaded_file in uploaded_files:

            # Temporary file path
            file_path = f"temp_{uploaded_file.name}"

            try:

                # -----------------------------------------
                # Save uploaded file
                # -----------------------------------------

                with open(file_path, "wb") as f:

                    f.write(uploaded_file.getbuffer())

                # -----------------------------------------
                # Ingest into RAG
                # -----------------------------------------

                with st.status(
                    f"Processing {uploaded_file.name}...",
                    expanded=True,
                ):

                    ingesting_into_rag(file_path)

                st.success(f"{uploaded_file.name} added to the knowledge base.")

            finally:

                # -----------------------------------------
                # Delete temporary file
                # -----------------------------------------

                if os.path.exists(file_path):

                    os.remove(file_path)

    # =====================================================
    # HANDLE USER MESSAGE
    # =====================================================

    if user_input:

        # ---------------------------------------------
        # Save user message to session history
        # ---------------------------------------------

        st.session_state["message_history"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # ---------------------------------------------
        # Display user message
        # ---------------------------------------------

        with st.chat_message("user"):

            st.text(user_input)

        # =================================================
        # LANGGRAPH CONFIG
        # =================================================

        CONFIG = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "chat_trace",
        }

        # =================================================
        # AI RESPONSE
        # =================================================

        with st.chat_message("assistant"):

            status_holder = {"box": None}

            def ai_only():

                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):

                    # =====================================
                    # TOOL MESSAGE
                    # =====================================

                    if isinstance(message_chunk, ToolMessage):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool",
                        )

                        if not status_holder["box"]:

                            status_holder["box"] = st.status(
                                f"Using {tool_name}",
                                expanded=True,
                            )

                        else:

                            status_holder["box"].update(
                                label=f"Using {tool_name}",
                                expanded=True,
                                state="running",
                            )

                    # =====================================
                    # AI MESSAGE
                    # =====================================

                    elif isinstance(message_chunk, AIMessage):

                        content = message_chunk.content

                        # ---------------------------------
                        # String content
                        # ---------------------------------

                        if isinstance(content, str):

                            if content:

                                yield content

                        # ---------------------------------
                        # List content
                        # ---------------------------------

                        elif isinstance(content, list):

                            texts = []

                            for item in content:

                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):

                                    texts.append(item.get("text", ""))

                            text = "".join(texts)

                            if text:

                                yield text

            # =================================================
            # STREAM AI RESPONSE
            # =================================================

            ai_msg = st.write_stream(ai_only())

            # =================================================
            # FINISH TOOL STATUS
            # =================================================

            if status_holder["box"]:

                status_holder["box"].update(
                    label="Finished using the tool",
                    state="complete",
                    expanded=False,
                )

        # =================================================
        # SAVE AI MESSAGE
        # =================================================

        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": ai_msg,
            }
        )


# =========================================================
# DEBUG: CURRENT LANGGRAPH STATE
# =========================================================

state = chatbot.get_state(
    config={"configurable": {"thread_id": st.session_state["thread_id"]}}
)
