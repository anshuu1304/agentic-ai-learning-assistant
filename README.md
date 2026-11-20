# 🤖 Agentic AI Learning Assistant

An AI-powered learning assistant built with **Python, LangGraph, Google Gemini, and Streamlit**. It combines conversational AI, document-based question answering, memory, and human-in-the-loop workflows to create an interactive learning experience.

## 🌐 Live Demo

**[Try Agentic AI Learning Assistant](https://agentic-ai-learning-assistant-f8dgriz7ppownnmyuuhxnk.streamlit.app/)**

## ✨ Features

- **AI-Powered Conversations:** Ask questions and interact with Google Gemini.
- **LangGraph Agent Workflow:** Manage AI interactions through a graph-based workflow.
- **Retrieval-Augmented Generation (RAG):** Ask questions about uploaded PDF documents using relevant document content.
- **PDF Document Upload:** Upload and process PDF files to build a searchable knowledge base.
- **Long-Term Memory:** Store and retrieve useful information across conversations.
- **Conversation History:** Maintain and revisit previous conversations.
- **Human-in-the-Loop (HITL):** Introduce human approval before proceeding with supported actions.
- **Streamlit Interface:** Interact with the assistant through a simple web application.

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Agent Framework | LangGraph |
| LLM | Google Gemini (`gemini-3.1-flash-lite`) |
| LLM Integration | LangChain |
| Embeddings | Gemini Embeddings |
| Vector Database | ChromaDB |
| Document Processing | PyPDF |
| Conversation Checkpointing | SQLite |
| Deployment | Streamlit Community Cloud |

## 🏗️ Architecture

```text
             User
               |
               v
        Streamlit Interface
               |
               v
        LangGraph Workflow
               |
       +-------+--------+
       |       |        |
       v       v        v
     Gemini   RAG     Memory
       |       |        |
       |    ChromaDB   SQLite
       |       |        |
       +-------+--------+
               |
               v
        AI Response
               |
               v
       Human Approval
       (When Required)
```

## 📂 Project Structure

```text
agentic-ai-learning-assistant/
│
├── app.py
├── agent.py
├── action.py
├── memory.py
├── rag.py
│
├── documents/
│   └── AIML:GenAI:AgenticAI.pdf
│
├── data/
│   └── Generated application data
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/anshuu1304/agentic-ai-learning-assistant.git

cd agentic-ai-learning-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Important:** Never upload your `.env` file or expose your API key on GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment configuration:

- Repository: `anshuu1304/agentic-ai-learning-assistant`
- Branch: `main`
- Main file: `app.py`
- Python version: `3.12`
- API configuration: Streamlit Secrets

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google Gemini API key |

For cloud deployment, configure the API key through Streamlit Cloud Secrets instead of committing it to the repository.

## ⚠️ Notes

- A sample PDF is included in the `documents/` directory.
- Users can upload additional PDF documents through the application.
- Local databases and uploaded data may not persist across Streamlit Cloud restarts or redeployments.
- Gemini API usage is subject to the applicable API quota and rate limits.

## 🎯 Project Objective

This project was developed to explore practical implementation of Agentic AI concepts, including LLM integration, LangGraph workflows, Retrieval-Augmented Generation, conversation memory, and human-in-the-loop interaction.

It demonstrates how these components can be combined into an interactive AI learning assistant.

---

**Built with Python, LangGraph, Google Gemini, and Streamlit.**