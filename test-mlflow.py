from llama_index.core import VectorStoreIndex, Document
import mlflow
import mlflow.llama_index
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Step 1: Enable autologging
mlflow.llama_index.autolog(log_traces=True)

# Step 2: Create a LlamaIndex model
documents = [
    Document(text="LlamaIndex integrates custom data with LLMs."),
    Document(text="This is a test document."),
]

index = VectorStoreIndex.from_documents(documents)

# Step 3: Log the model
with mlflow.start_run():
    model_info = mlflow.llama_index.log_model(
        llama_index_model=index,
        artifact_path="llama_index_model",
        engine_type="query",  # Specify engine type
        registered_model_name="LlamaIndexModelDemo",
    )
    
    # Step 4: Load the model
    loaded_model = mlflow.llama_index.load_model(model_info.model_uri)
    
    # Step 5: Create a QueryEngine
    query_engine = loaded_model.as_query_engine()
    
    # Step 6: Perform inference
    eval_data = pd.DataFrame(
        {
            "inputs": [
                "What does LlamaIndex do?",
                "What is this document about?",
            ],
            "ground_truth": [
                "LlamaIndex integrates custom data with LLMs.",
                "This is a test document.",
            ],
        }
    )

    # Define a custom function to generate predictions using the query engine
    def query_engine_predict(inputs):
        return [query_engine.query(input_text).response for input_text in inputs["inputs"]]
    
    # Step 7: Evaluate the model using MLflow LLM metrics
    results = mlflow.evaluate(
        model=query_engine_predict,  # Pass the custom query engine function
        data=eval_data,
        targets="ground_truth",
        model_type="question-answering",  # Use pre-defined question-answering metrics
        extra_metrics=[
            mlflow.metrics.genai.answer_correctness(),  # Built-in LLM-as-a-Judge metric
            mlflow.metrics.latency(),  # Evaluate latency
        ],
    )
    
    # Step 8: Print and log evaluation results
    print("Aggregated Metrics:", results.metrics)
    eval_table = results.tables["eval_results_table"]
    print("Evaluation Results Table:\n", eval_table)

    # Log evaluation results to MLflow
    for metric_name, metric_value in results.metrics.items():
        mlflow.log_metric(metric_name, metric_value)
