from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENROUTER_API_KEY=os.environ.get('openrouter_api_key') 
openai_base = "https://openrouter.ai/api/v1"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
#os.environ["OPENROUTER_API_KEY"]="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"


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

llm = ChatOpenAI(
    openai_api_base=openai_base,
    #openai_api_key = "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",
    openai_api_key=OPENROUTER_API_KEY,
    model_name="deepseek/deepseek-r1-0528:free",
    temperature=0.0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)