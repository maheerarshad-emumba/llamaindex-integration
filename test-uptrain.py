from llama_index.core import VectorStoreIndex, Document
from llama_index.callbacks.uptrain.base import UpTrainCallbackHandler
from dotenv import load_dotenv
import os
from getpass import getpass

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key: ")

# Create an index
documents = [
    Document(text="LlamaIndex integrates custom data with LLMs."),
    Document(text="This is a test document."),
]
index = VectorStoreIndex.from_documents(documents)

# Integrate UpTrain callback handler
uptrain_handler = UpTrainCallbackHandler(
    key_type="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    project_name="uptrain_llamaindex",
)
query_engine = index.as_query_engine(callbacks=[uptrain_handler])

# Query the engine
query = "What does LlamaIndex do?"
response = query_engine.query(query)
print(f"Query: {query}\nResponse: {response}\n")

# Log evaluation manually (if UpTrain metrics are unavailable)
# This can be extended for more detailed metrics
expected_response = "LlamaIndex integrates custom data with LLMs."
print("Evaluation Results:")
if expected_response.lower() in str(response).lower():
    print("Factual Accuracy: 1.0")
    print("Response Completeness: 1.0")
else:
    print("Factual Accuracy: 0.0")
    print("Response Completeness: 0.0")
