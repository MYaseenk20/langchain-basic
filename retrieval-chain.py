from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitted_doc = splitter.split_documents(docs)
    return splitted_doc


def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(documents=docs,embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.7,
        verbose=True
    )

    prompt = ChatPromptTemplate.from_template(
        """
            Answer the user's question about Android Activity Lifecycle:
            Context : {context}
            Question: {input}    
    """)

    # chain = prompt | model

    chain = create_stuff_documents_chain(
        llm = model,
        prompt=prompt
    )
    
    #search_kwargs={"k": 3}
    retriever = vectorStore.as_retriever()
    chainRetriever = create_retrieval_chain(retriever,combine_docs_chain=chain)
    
    return chainRetriever
    
    


docs = get_documents_from_web("https://developer.android.com/guide/components/activities/activity-lifecycle")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)


response = chain.invoke(
    {"input":"Explain the Android Activity Lifecycle"}
    )

print(response["answer"])