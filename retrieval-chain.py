from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    )
    splitted_doc = splitter.create_documents(docs)
    return splitted_doc

def create_db(docs):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorStore = FAISS.from_documents(documents=docs,embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOllama(
        model="llama3.2:3b", 
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(
        """
            Answer the user's question about Android Activity Lifecycle:
            Context : {context}
            Question: {input}    
    """)
    
    chain = create_stuff_documents_chain(llm=model,prompt=prompt)
    
    retriever = vectorStore.as_retriever()
    
    chainRetriever = create_retrieval_chain(retriever=retriever,combine_docs_chain=chain)
    
    return chainRetriever


docs = get_documents_from_web("https://developer.android.com/guide/components/activities/activity-lifecycle")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input":"what is onResume in lifecycle"
})    

print(response["answer"])
