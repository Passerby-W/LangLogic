from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langsmith.run_helpers import traceable
import os
from utils import get_config
from manager import OpenAIManager
from typing import Tuple
from embedding import OpenAIEmbedding
from llm import OpenAILLM


os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = get_config("KEY_LANGSMITH")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "web_rag"

manager = OpenAIManager()
llm = OpenAILLM()
embedding = OpenAIEmbedding()

import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings


@traceable(run_type="llm", name="openai.ChatCompletion.create")
def chat_model(*args, **kwargs):
    return manager.chat(*args, **kwargs)


@traceable(run_type="tool")
def load_docs(urls: Tuple):
    loader = WebBaseLoader(web_paths=urls)
    docs = loader.load()
    return docs


@traceable(run_type="tool")
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    docs = text_splitter.split_documents(docs)
    return docs


@traceable(run_type="tool")
def vectorize_docs(docs):
    db = Chroma.from_documents(documents=docs, embedding=embedding, collection_name="text")
    return db


@traceable(run_type="tool")
def find_docs(query, db):
    ans = db.similarity_search_by_vector(embedding=embedding.embed_query(query), k=5)
    return ans


@traceable(run_type="tool")
def combine_docs(docs):
    content = "\n".join(doc.page_content for doc in docs)
    return content


@traceable(run_type="tool")
def prepare_data(query, content):
    data = {
        "model": "gpt-4-1106-preview",
        "messages": [
            {
                "role": "system",
                "content": "你是一个仔细并且准确的总结答复助手，请根据用户给的背景材料来回答用户提问。"
            },
            {
                "role": "user",
                "content": f"根据我所找到的材料：{content}，我想知道{query}的答案。如果材料中没有足够的信息回答这个问题，则不要胡编乱造答案。"
            },
        ]
    }
    return data


@traceable(run_type="tool")
def get_ans(completion):
    return completion.choices[0].message.content


@traceable(run_type="chain")
def preprocessing_chain(urls):
    docs = load_docs(urls)
    docs = split_docs(docs)
    db = vectorize_docs(docs)
    return db


@traceable(run_type="chain")
def retrial_chain(query, db):
    docs_find = find_docs(query, db)
    content = combine_docs(docs_find)
    return content


@traceable(run_type="chain")
def qa_chain(query, content):
    data = prepare_data(query, content)
    completion = chat_model(**data)
    ans = get_ans(completion)
    return ans


@traceable(run_type="chain")
def main_chain(urls: Tuple, query) -> str:
    db = preprocessing_chain(urls)
    content = retrial_chain(query, db)
    ans = qa_chain(query, content)
    return ans


a = main_chain(("https://zhuanlan.zhihu.com/p/670803806",), "通俗易懂的给我解释下这篇文章的公式")
print(a)
