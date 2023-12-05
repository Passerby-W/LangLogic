import bs4
from langchain import hub
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from embed import OpenAIEmbeddings

loader = WebBaseLoader(web_paths=("https://www.zhihu.com/question/633391682/answer/3314363232"))

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
embedding = OpenAIEmbeddings()

# load it into Chroma
db = Chroma.from_documents(documents=splits, embedding=embedding, collection_name="text")

query = "南京是第几梯队"
ans = db.similarity_search_by_vector(embedding=embedding.embed_query(query), k=3)


ans = [d for d]
rel = []
