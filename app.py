from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Cassandra
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
print('api',HUGGINGFACEHUB_API_TOKEN)

# Initialize models and vector store (reuse your existing setup)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

llm_model_name = "facebook/opt-350m"
llm = HuggingFaceHub(repo_id=llm_model_name, model_kwargs={"temperature": 0.7}, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN ,database_id=ASTRA_DB_ID)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,  # Connect your AstraDB session here
    keyspace=None  # Provide your keyspace here
)

# Initialize FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Define endpoint for querying
@app.post('/query')
def query_vector_store(request: QueryRequest):
    query_text = request.question
    if query_text:
        vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
        answer = vector_index.query(query_text, llm=llm).strip()
        return {'question': query_text, 'answer': answer}
    else:
        raise HTTPException(status_code=400, detail="No question provided")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
