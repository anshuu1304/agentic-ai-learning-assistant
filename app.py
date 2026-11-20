
import streamlit as st
import uuid
import tempfile
import os
from rag import load_folder_documents
from agent import set_vector_store
import agent as agent_module
from rag import list_uploaded_pdfs, delete_pdf
import json
from agent import agent, Command



# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


CONVERSATIONS_FILE = "data/conversations.json"


def load_conversations():
    if not os.path.exists(CONVERSATIONS_FILE):
        return {}

    with open(CONVERSATIONS_FILE, "r") as file:
        return json.load(file)


def save_conversations(conversations):
    os.makedirs("data", exist_ok=True)

    with open(CONVERSATIONS_FILE, "w") as file:
        json.dump(conversations, file, indent=4)


def save_current_conversation():
    thread_id = st.session_state.thread_id
    messages = st.session_state.messages

    if not messages:
        return

    first_question = next(
        (msg["content"] for msg in messages if msg["role"] == "user"),
        "New Conversation"
    )

    conversations = st.session_state.conversations

    conversations[thread_id] = {
        "title": first_question[:35],
        "messages": messages,
        "agent_messages": st.session_state.agent_messages
    }

    save_conversations(conversations)

    st.session_state.conversations = conversations


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

    .stApp {
        background-color: #f8f9fb;
        color: #202124;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] p {
        color: #4b5563;
    }

    .main-title {
        font-size: 27px;
        font-weight: 650;
        color: #202124;
        letter-spacing: -0.7px;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 28px;
    }

    .section-label {
        color: #6b7280;
        font-size: 11px;
        font-weight: 650;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 22px;
        margin-bottom: 10px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 9px;
        padding: 15px;
        margin-bottom: 10px;
    }

    .info-title {
        font-size: 14px;
        font-weight: 600;
        color: #202124;
        margin-bottom: 5px;
    }

    .info-text {
        font-size: 12px;
        color: #6b7280;
        line-height: 1.5;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
    }

    [data-testid="stChatInput"] {
        border-radius: 12px;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff;
    }

    .stButton button {
        border-radius: 7px;
        font-weight: 500;
        border: 1px solid #d1d5db;
    }

    .stButton button[kind="primary"] {
        background-color: #202124;
        color: white;
        border: 1px solid #202124;
    }

    .stButton button[kind="primary"]:hover {
        background-color: #3f4145;
        border-color: #3f4145;
    }

    .status-text {
        font-size: 12px;
        color: #15803d;
        font-weight: 500;
    }

    footer {
        visibility: hidden;
    }

    /* Global text colour */
    .stApp,
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stMarkdown,
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] p,
    [data-testid="stSidebar"] p {
        color: #111827 !important;
    }
    
    /* Chat input text */
    .stChatInput textarea {
        color: #111827 !important;
        background-color: #ffffff !important;
    }
    
    /* Input placeholder */
    .stChatInput textarea::placeholder {
        color: #6b7280 !important;
    }

    /* Sidebar button text */
    [data-testid="stSidebar"] button {
        color: white !important;
    }
    
    /* File uploader button text */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        color: white !important;
    }


    /* Force white text on sidebar buttons */
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stButton button span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button span {
        color: #ffffff !important;
    }

    /* Sidebar buttons: force white text */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    

    /* Chat input container */
    [data-testid="stChatInput"] {
        width: 100% !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
    }
    
    /* Chat input field */
    [data-testid="stChatInput"] textarea {
        min-height: 55px !important;
        font-size: 16px !important;
        padding: 14px 16px !important;
        color: #111827 !important;
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Remove focus border/shadow */
    [data-testid="stChatInput"] textarea:focus {
        border: 1px solid #d1d5db !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Placeholder */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #6b7280 !important;
        font-size: 15px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = False

if "approval_question" not in st.session_state:
    st.session_state.approval_question = ""

if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()


# ---------------- AGENT CONFIG ----------------

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ---------------- HELPER FUNCTIONS ----------------

def process_result(result):

    if "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0]

        st.session_state.pending_approval = True

        st.session_state.approval_question = getattr(
            interrupt_data,
            "value",
            str(interrupt_data)
        )

    else:

        st.session_state.pending_approval = False
        st.session_state.approval_question = ""

        answer = result.get(
            "answer",
            "I could not generate a response."
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        if "messages" in result:
            st.session_state.agent_messages = result["messages"]

    save_current_conversation()



def start_new_conversation():

    save_current_conversation()

    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.agent_messages = []
    st.session_state.pending_approval = False
    st.session_state.approval_question = ""


def handle_approval(decision):

    try:

        result = agent.invoke(
            Command(resume=decision),
            config=config
        )

        process_result(result)

    except Exception as error:

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Something went wrong: {error}"
        })

        st.session_state.pending_approval = False


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        '<div class="main-title">Agentic AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Learning Assistant</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "＋  New conversation",
        use_container_width=True
    ):
        start_new_conversation()
        st.rerun()

    st.subheader("Conversation History")

    for chat_id, chat_data in st.session_state.conversations.items():
    
        if st.button(
            chat_data["title"],
            key=f"chat_{chat_id}",
            use_container_width=True
        ):
            st.session_state.thread_id = chat_id
            st.session_state.messages = chat_data["messages"]
            st.session_state.agent_messages = chat_data.get("agent_messages", [])
            st.session_state.pending_approval = False
            st.session_state.approval_question = ""
    
            st.rerun()    

    st.markdown(
        '<div class="section-label">Capabilities</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <div class="info-title">General Assistant</div>
        <div class="info-text">
            Ask questions and get answers using Gemini.
        </div>
    </div>

    <div class="info-card">
        <div class="info-title">Document Search</div>
        <div class="info-text">
            Ask questions about your uploaded PDF documents using RAG.
        </div>
    </div>

    <div class="info-card">
        <div class="info-title">Action Approval</div>
        <div class="info-text">
            Review and approve actions before they are executed.
        </div>
    </div>

    <div class="info-card">
        <div class="info-title">Conversation Memory</div>
        <div class="info-text">
            Uses short-term and long-term memory to retain relevant information.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">System Status</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="status-text">● Assistant ready</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-text">Powered by LangGraph and Gemini</div>',
        unsafe_allow_html=True
    )

    st.markdown(
    '<div class="section-label">Upload Documents</div>',
    unsafe_allow_html=True
    )
    
    uploaded_files = st.file_uploader(
        "Select PDF Folder",
        type=["pdf"],
        accept_multiple_files="directory"
    )
    
    if st.button("Process Folder", use_container_width=True):
        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
    
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
    
                try:
                    new_vector_store = load_folder_documents(temp_dir)
                    set_vector_store(new_vector_store)
                    st.success(f"{len(uploaded_files)} PDFs processed!")
                except Exception as error:
                    st.error(str(error))
        else:
            st.warning("Please select a folder.")

    st.markdown(
    '<div class="section-label">Uploaded PDFs</div>',
    unsafe_allow_html=True
    )
    
    try:
        pdf_names = list_uploaded_pdfs(agent_module.vector_store)
    
        if pdf_names:
            for filename in pdf_names:
                col1, col2 = st.columns([3, 1])
    
                with col1:
                    st.markdown(f"📄 {filename}")
    
                with col2:
                    if st.button("Delete", key=f"delete_{filename}"):
                        delete_pdf(agent_module.vector_store, filename)
                        st.success(f"{filename} deleted!")
                        st.rerun()
        else:
            st.caption("No PDFs uploaded yet.")
    
    except Exception as error:
        st.error(str(error))        


# ---------------- MAIN CHAT AREA ----------------

st.markdown(
    '<div class="main-title">Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask a question, explore your documents, or request an action.</div>',
    unsafe_allow_html=True
)


# ---------------- WELCOME SCREEN ----------------

if not st.session_state.messages:

    st.markdown("### How can I help you today?")

    st.markdown(
        "Start a conversation by asking a question below."
    )

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Ask a general question</div>
            <div class="info-text">
                Learn about Python, AI, machine learning and other topics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">Explore your documents</div>
            <div class="info-text">
                Ask questions about information in your uploaded PDF.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------- HUMAN APPROVAL ----------------

if st.session_state.pending_approval:

    with st.chat_message("assistant"):

        st.markdown("**Action requires your approval**")

        st.write(st.session_state.approval_question)

        st.caption(
            "The action will only execute if you approve it."
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Approve",
                type="primary",
                use_container_width=True,
                key="approve_action"
            ):

                handle_approval("yes")
                st.rerun()

        with col2:

            if st.button(
                "Reject",
                use_container_width=True,
                key="reject_action"
            ):

                handle_approval("no")
                st.rerun()


# ---------------- CHAT INPUT ----------------

if not st.session_state.pending_approval:

    question = st.chat_input(
        "Message your assistant..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Thinking..."):

            try:

                result = agent.invoke(
                    {
                        "question": question,
                        "answer": "",
                        "messages": st.session_state.agent_messages,
                        "action": "",
                        "approval": ""
                    },
                    config=config
                )

                process_result(result)

            except Exception as error:

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Something went wrong: {error}"
                })

        st.rerun()
