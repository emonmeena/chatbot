# File available in the profile
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import pinecone
from langchain.vectorstores import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
print(openai_api_key)

folder_name = "biobotdocs"
file_list = os.listdir(folder_name)


pinecone.init(
    api_key='b1e8aa5b-abdd-4158-9a91-fff36d29aaa3',
    environment='gcp-starter'
)
index = pinecone.Index('biobot')


for file_name in file_list:

    loader = PyPDFLoader(folder_name + "/" + file_name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(
        deployment="text-embedding-ada-002", model="text-embedding-ada-002", chunk_size=1)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name='biobot')

query = "what is prepaire labs?"
docs = docsearch.similarity_search(query)
# print(docs)
