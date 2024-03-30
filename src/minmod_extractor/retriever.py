import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class Retriever(object):
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.embeddings_model = OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )

    def run(self):
        docs = self.load_docs()

    def load_docs(self, path) -> list[Document]:
        """Load docs at the file path"""
        loader = TextLoader(path)
        docs = loader.load()

        return docs

    def split_text(self, docs: list[Document]) -> list[Document]:
        splitted_docs = self.text_splitter.split_documents(docs)
        return splitted_docs

    def embed_docs(self, docs: list[Document]) -> list[list[float]]:
        embedded_docs = self.embeddings_model.embed_documents(docs)
        return embedded_docs

    def embed_query():
        pass
