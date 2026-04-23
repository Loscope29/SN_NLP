import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langsmith import traceable

load_dotenv()  # Charge les variables d'environnement depuis le fichier .env

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")  # Clé pour Google Gemini


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_text(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def create_vector_store(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return Chroma.from_documents(chunks, embeddings)


def create_retriever(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": 5})


def llm_setup():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=None # Il la cherchera par défaut dans st.secrets ou os.environ
    )

    system_prompt = (
        "Tu es un assistant expert. Utilise les fragments de contexte suivants "
        "pour répondre à la question. Si tu ne connais pas la réponse, dis-le.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
    ])
    
    return llm, prompt


def qa_chain(retriever, llm, prompt):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


@traceable("RAG-Process")
def main(file_path, question):
    documents = load_pdf(file_path)
    chunks = split_text(documents)
    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store)
    llm, prompt = llm_setup()
    qa = qa_chain(retriever, llm, prompt)
    result = qa.invoke({"input": question})
  
    return result


  

    


