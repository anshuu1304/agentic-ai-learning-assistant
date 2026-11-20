from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from langgraph.graph import StateGraph , START , END
from rag import load_and_store_documents , generate_rag_answers
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import json
import os

from langgraph.types import interrupt , Command




load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
os.makedirs(DATA_DIR, exist_ok=True)

LTM_FILE = os.path.join(DATA_DIR, "LTM.json")

def load_memory():

    if not os.path.exists(LTM_FILE):
        return {}

    with open(LTM_FILE , "r") as file:
        return json.load(file)


def save_memory(key , value):

    memory = load_memory()

    memory[key] = value

    with open(LTM_FILE , "w") as file:
        json.dump(memory , file , indent=4)




def extract_user_facts(question):

    prompt = f"""
    Extract important information explicitly shared
    by the user from the following message.

    Extract stable, non-sensitive facts such as:
    name, interests, and preferences.

    Return only valid JSON.
    If no useful information is found, return {{}}.

    User message: {question}
    """

    response= llm.invoke(prompt)
    facts = json.loads(response.content[0]["text"])

    return facts




def summarize_conversation(messages):

    prompt = f"""
    Summarize the following conversation.
    Preserve important facts, topics and user preferences.
    Keep the summary short and clear.

    Conversation:
    {messages}
    """
    response = llm.invoke(prompt)

    summary = response.content[0]["text"]
    return summary




def generate_summary(messages):

    conversations = "\n".join(
        f"{message['role']}:{message['content']}"
        for message in messages
    )
    response = llm.invoke(f"summarize this conversation \n{conversations} breifly")
    return response.content



class AgentState(TypedDict):
    question:str
    answer:str
    messages:list
    action : str
    approval: str



def route_question(state: AgentState):

    question = state["question"]

    prompt = f"""
    Classify the user's request into one category:

    - document: Questions about uploaded PDFs, notes or files.
    - action: Requests to perform an action, such as creating a reminder.
    - general: All other questions.

    User request: {question}

    Return only one word: document, action or general.
    """

    response = llm.invoke(prompt)

    category = response.content[0]["text"].strip().lower()

    if "document" in category:
        return "document"

    if "action" in category:
        return "action"

    return "general"




def general_answer(state:AgentState):

    messages = state["messages"]

    messages.append({
            "role" : "user",
            "content" : state["question"]
    })

    trimmed_messages = messages[-10:]
    print("Original Messages:", len(messages))
    print("Trimmed Messages:", len(trimmed_messages))

    facts = extract_user_facts(state["question"])

    for key,value in facts.items():
        save_memory(key,value)

    memory = load_memory()

    conversation_summary = memory.get("conversation_summary","")

    memory_context = f"""
    You are a helpful personal AI assistant.
    
    Use the following saved user information
    and previous conversation summary when relevant.
    
    Saved User Information:
    {memory}
    
    Previous Conversation Summary:
    {conversation_summary}
    """

    response = llm.invoke(memory_context+"\n"+str(trimmed_messages))

    answer = response.content[0]["text"]

    messages.append({
        "role":"assistant",
        "content": answer
    })

    if len(messages) >= 10 and len(messages)%10 == 0:
    
            summary = summarize_conversation(messages)
            save_memory("conversation_summary", summary)
            print("Conversation Summary Updated!")

    return {
        "answer" : answer,
        "messages":messages
    }




def approval_node(state: AgentState):

    action = state["question"]

    decision = interrupt(
        f"Do you approve this action: {action}?"
    )

    return {
        "action": action,
        "approval": decision
    }




def execute_action(state:AgentState):

    reminder_files = os.path.join(DATA_DIR, "reminders.json")

    if not os.path.exists(reminder_files):
        with open(reminder_files, "w") as file:
            json.dump([], file)

    with open(reminder_files , "r") as file:
        reminders = json.load(file)

    reminders.append({
        "remainder" : state["action"],
        "status":"approved"
    })

    with open(reminder_files,"w") as file:
        json.dump(reminders , file , indent=4)

    print("Action Approved!")
    print("Reminder Saved Successfully!")

    return {
        "answer": "Your reminder has been approved and saved."
    }




def check_approval(state:AgentState):
    if state["approval"] == "yes":
        return "execute"
    else:
        return "reject"



def reject_action(state: AgentState):

    return {
        "answer": "Action rejected by human. Nothing was executed."
    }    




def document_answer(state:AgentState):

    answer = generate_rag_answers(
        vector_store,
        state["question"],
        llm
    )
    return {"answer":answer}





graph_builder = StateGraph(AgentState)

pdf_path = os.path.join(DOCUMENTS_DIR, "AIML:GenAI:AgenticAI.pdf")
vector_store = load_and_store_documents(pdf_path)

def set_vector_store(new_vector_store):
    global vector_store
    vector_store = new_vector_store

    

graph_builder.add_node("general_answer",general_answer)
graph_builder.add_node("document_answer",document_answer)

graph_builder.add_conditional_edges(
    START,
    route_question,
    {
        "general":"general_answer",
        "document":"document_answer",
        "action":"hitl"
    }
)

graph_builder.add_edge("general_answer",END)
graph_builder.add_edge("document_answer",END)

conn = sqlite3.connect("data/agent_memory.db" , check_same_thread=False)

checkpointer = SqliteSaver(conn)
checkpointer.setup()

# agent = graph_builder.compile(checkpointer=checkpointer)



hitl_builder = StateGraph(AgentState)

hitl_builder.add_node("approval", approval_node)
hitl_builder.add_node("execute", execute_action)
hitl_builder.add_node("reject", reject_action)

hitl_builder.set_entry_point("approval")

hitl_builder.add_conditional_edges(
    "approval",
    check_approval,
    {
        "execute": "execute",
        "reject": "reject"
    }
)

hitl_builder.add_edge("execute", END)
hitl_builder.add_edge("reject", END)

hitl_graph = hitl_builder.compile(checkpointer=checkpointer)

graph_builder.add_node("hitl", hitl_graph)

agent = graph_builder.compile(checkpointer=checkpointer)




if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "main_agent_test"
        }
    }

    question = input("How can I help you? ")

    result = agent.invoke(
        {
            "question": question,
            "answer": "",
            "messages": [],
            "action": "",
            "approval": ""
        },
        config=config
    )

    if "__interrupt__" in result:

        decision = input("\nApprove this action? (yes/no): ")

        result = agent.invoke(
            Command(resume=decision),
            config=config
        )

        print("\nFinal Response:")
        print(result["answer"])

    print("\nAgent Response:")
    print(result)