from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS

pdf_loader=PyPDFLoader('Lecture4.pdf',extract_images=True)
chunks=pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10))

embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_english-base')

vector_db=FAISS.from_documents(chunks,embeddings)
vector_db.save_local('Lecture4.faiss')

print('faiss saved!')



