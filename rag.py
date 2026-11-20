from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)


def extract_text(pdf_path):

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()or ""
    return text




def split_text(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def load_and_store_documents(pdf_path):
    text = extract_text(pdf_path)
    chunks = split_text(text)
    vector_store = store_chunks(chunks)

    return vector_store



def load_folder_documents(folder_path):
    all_chunks = []
    all_metadatas = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)

            text = extract_text(pdf_path)

            if text.strip():
                chunks = split_text(text)
                all_chunks.extend(chunks)

                for chunk in chunks:
                    all_metadatas.append({"source": filename})

    if not all_chunks:
        raise ValueError("No readable PDF documents found.")

    vector_store = store_chunks(all_chunks, all_metadatas)

    return vector_store



def store_chunks(chunks , metadatas = None):

    vector_store = Chroma(
        collection_name = "agentic_ai_documents",
        embedding_function = embeddings,
        persist_directory = "data/chroma_db"
    )
    exisiting_ids = vector_store.get()["ids"]
    if exisiting_ids:
        vector_store.delete(ids=exisiting_ids)

    vector_store.add_texts(chunks, metadatas=metadatas)  
    return vector_store


def list_uploaded_pdfs(vector_store):
    data = vector_store.get(include=["metadatas"])

    pdf_names = set()

    for metadata in data["metadatas"]:
        if metadata and "source" in metadata:
            pdf_names.add(metadata["source"])

    return sorted(pdf_names)




def delete_pdf(vector_store, filename):
    results = vector_store.get(
        where={"source": filename}
    )

    if results["ids"]:
        vector_store.delete(ids=results["ids"])
        return True

    return False



def search_documents(vector_store , question):

    results = vector_store.similarity_search(
        query = question,
        k =3
    )
    return results



def generate_rag_answers(vector_store , question , model):

    results = search_documents(vector_store , question)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
    Answer the question {question} using only the provided context.
    If the answer is not available in the context {context},
    clearly say that you don't know.
    """
    response = model.invoke(prompt)
    return response.content[0]["text"]
    










if __name__ == "__main__":

    pdf_path = "/Users/pranavshukla/Downloads/data science/AgenticAI/LangGraph/agentic-ai-learning/documents/AIML:GenAI:AgenticAI.pdf"

    extracted_text = extract_text(pdf_path)
    chunks = split_text(extracted_text)
    sample_text = chunks[0]
    # vector = embeddings.embed_query(sample_text)
    vector_store = store_chunks(chunks)

    question = "What is Agentic AI?"
    results = search_documents(vector_store, question)
    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(results):
        print(f"\nChunk {i + 1}:")
        print(doc.page_content[:500])

    # print("Total Characters:", len(extracted_text))
    print("Total Chunks:", len(chunks))
    # print("\nFirst Chunk:\n")
    # print(chunks[0])
    # print("Embedding generated successfully!")
    # print("Vector dimensions:", len(vector))
    # print("First 5 values:", vector[:5])
    # print("Chunks stored successfully!")
    # print("Total stored chunks:", vector_store._collection.count())