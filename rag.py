from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import os 

embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_english-base')



vector_db=FAISS.load_local('Lecture4.faiss',embeddings)

retriever=vector_db.as_retriever(search_kwargs={"k":5})

chat=ChatOpenAI(
    openai_api_key="key",
)

system_prompt=SystemMessagePromptTemplate.from_template('You are a helpful assistant.')
user_prompt=HumanMessagePromptTemplate.from_template('''
Answer the question based only on the following context:

{context}

Question: {query}
''')
full_chat_prompt=ChatPromptTemplate.from_messages([system_prompt,MessagesPlaceholder(variable_name="chat_history"),user_prompt])


# Chat chain
chat_chain={
        "context": itemgetter("query") | retriever,
        "query": itemgetter("query"),
        "chat_history":itemgetter("chat_history"),
    }|full_chat_prompt|chat


chat_history=[]
while True:
    query=input('query:')
    response=chat_chain.invoke({'query':query,'chat_history':chat_history})
    chat_history.extend((HumanMessage(content=query),response))
    print(response.content)
    chat_history=chat_history[-20:]
