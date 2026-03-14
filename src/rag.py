from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def build_vectorstore(project_root):
    docs_path = project_root / "docs"
    loader = DirectoryLoader(str(docs_path), glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 5):
    docs = vectorstore.similarity_search(query, k=k)
    return docs