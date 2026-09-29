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

from langgraph.types import Command

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
# HITL FUNCTIONS
# =========================================================


def get_pending_interrupt(thread_id):

    config = {"configurable": {"thread_id": thread_id}}

    state = chatbot.get_state(config=config)

    for task in state.tasks:

        interrupts = getattr(task, "interrupts", ())

        if interrupts:

            interrupt_obj = interrupts[0]

            return {
                "id": getattr(interrupt_obj, "id", None),
                "value": getattr(interrupt_obj, "value", interrupt_obj),
            }

    return None


def resume_after_human_decision(approved):

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "human_approval_resume",
    }

    return chatbot.stream(
        Command(resume={"approved": approved}),
        config=CONFIG,
        stream_mode="messages",
    )


# =========================================================
# MESSAGE EXTRACTION
# =========================================================


def extract_ai_text(message):

    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        texts = []

        for item in content:

            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))

        return "".join(texts)

    return str(content)


# =========================================================
# STREAM AI RESPONSE
# =========================================================


def stream_ai_response(stream):

    for message_chunk, metadata in stream:

        # =============================================
        # TOOL MESSAGE
        # =============================================

        if isinstance(message_chunk, ToolMessage):

            tool_name = getattr(
                message_chunk,
                "name",
                "tool",
            )

            if not st.session_state.get("status_box"):

                st.session_state["status_box"] = st.status(
                    f"Using {tool_name}",
                    expanded=True,
                )

            else:

                st.session_state["status_box"].update(
                    label=f"Using {tool_name}",
                    state="running",
                    expanded=True,
                )

        # =============================================
        # AI MESSAGE
        # =============================================

        elif isinstance(message_chunk, AIMessage):

            text = extract_ai_text(message_chunk)

            if text:
                yield text


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================


if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_id()


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = gettig_all_threads()


if "status_box" not in st.session_state:

    st.session_state["status_box"] = None


# Add current thread
add_thread_id(st.session_state["thread_id"])


# =========================================================
# PAGE
# =========================================================


st.title("Chat Bot")

st.write(f"Current chat: " f"{st.session_state['thread_id']}")


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

            # =========================================
            # USER MESSAGE
            # =========================================

            if isinstance(message, HumanMessage):

                temp_messages.append(
                    {
                        "role": "user",
                        "content": message.content,
                    }
                )

            # =========================================
            # AI MESSAGE
            # =========================================

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
# CHECK FOR EXISTING HITL INTERRUPT
# =========================================================

pending_interrupt = get_pending_interrupt(st.session_state["thread_id"])


if pending_interrupt:

    interrupt_value = pending_interrupt["value"]

    st.warning("Human approval is required.")

    # The value comes from:
    #
    # interrupt(
    #     f"Do you want to send..."
    # )
    #
    # Therefore it is normally a string.

    st.info(str(interrupt_value))

    col1, col2 = st.columns(2)

    # =====================================================
    # APPROVE
    # =====================================================

    with col1:

        approve = st.button(
            "✅ Approve",
            key=(f"approve_" f"{pending_interrupt['id']}"),
            use_container_width=True,
        )

    # =====================================================
    # REJECT
    # =====================================================

    with col2:

        reject = st.button(
            "❌ Reject",
            key=(f"reject_" f"{pending_interrupt['id']}"),
            use_container_width=True,
        )

    # =====================================================
    # HANDLE DECISION
    # =====================================================

    if approve or reject:

        approved = approve

        st.session_state["status_box"] = None

        with st.chat_message("assistant"):

            resumed_stream = resume_after_human_decision(approved)

            resumed_text = st.write_stream(stream_ai_response(resumed_stream))

        # =================================================
        # SAVE RESUMED AI RESPONSE
        # =================================================

        if resumed_text:

            st.session_state["message_history"].append(
                {
                    "role": "assistant",
                    "content": resumed_text,
                }
            )

        # =================================================
        # FINISH TOOL STATUS
        # =================================================

        if st.session_state.get("status_box"):

            st.session_state["status_box"].update(
                label="Finished",
                state="complete",
                expanded=False,
            )

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

            file_path = f"temp_{uploaded_file.name}"

            try:

                # =========================================
                # SAVE FILE
                # =========================================

                with open(file_path, "wb") as f:

                    f.write(uploaded_file.getbuffer())

                # =========================================
                # INGEST INTO RAG
                # =========================================

                with st.status(
                    (f"Processing " f"{uploaded_file.name}..."),
                    expanded=True,
                ):

                    ingesting_into_rag(file_path)

                st.success((f"{uploaded_file.name} " "added to the knowledge base."))

            finally:

                # =========================================
                # DELETE TEMP FILE
                # =========================================

                if os.path.exists(file_path):

                    os.remove(file_path)

    # =====================================================
    # HANDLE USER MESSAGE
    # =====================================================

    if user_input:

        # ================================================
        # SAVE USER MESSAGE
        # ================================================

        st.session_state["message_history"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # ================================================
        # DISPLAY USER MESSAGE
        # ================================================

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

        st.session_state["status_box"] = None

        # =================================================
        # AI RESPONSE
        # =================================================

        with st.chat_message("assistant"):

            ai_msg = st.write_stream(
                stream_ai_response(
                    chatbot.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    )
                )
            )

        # =================================================
        # CHECK IF GRAPH PAUSED
        # =================================================

        pending_interrupt = get_pending_interrupt(st.session_state["thread_id"])

        if pending_interrupt:

            # =============================================
            # HITL UI
            # =============================================

            interrupt_value = pending_interrupt["value"]

            st.warning("Human approval is required.")

            st.info(str(interrupt_value))

            col1, col2 = st.columns(2)

            # =============================================
            # APPROVE BUTTON
            # =============================================

            with col1:

                approve = st.button(
                    "✅ Approve",
                    key=("approve_new_" f"{pending_interrupt['id']}"),
                    use_container_width=True,
                )

            # =============================================
            # REJECT BUTTON
            # =============================================

            with col2:

                reject = st.button(
                    "❌ Reject",
                    key=("reject_new_" f"{pending_interrupt['id']}"),
                    use_container_width=True,
                )

            # =============================================
            # HANDLE DECISION
            # =============================================

            if approve or reject:

                approved = approve

                with st.chat_message("assistant"):

                    resumed_stream = resume_after_human_decision(approved)

                    resumed_text = st.write_stream(stream_ai_response(resumed_stream))

                if resumed_text:

                    st.session_state["message_history"].append(
                        {
                            "role": "assistant",
                            "content": resumed_text,
                        }
                    )

                st.rerun()

        else:

            # =================================================
            # FINISH TOOL STATUS
            # =================================================

            if st.session_state.get("status_box"):

                st.session_state["status_box"].update(
                    label="Finished using the tool",
                    state="complete",
                    expanded=False,
                )

            # =================================================
            # SAVE AI MESSAGE
            # =================================================

            if ai_msg:

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
