# LlamaIndex Integration PoC

This repository demonstrates the integration of **LlamaIndex** with **MLflow** and **UpTrain** to manage, track, and evaluate AI workflows.

## Overview
- **LlamaIndex**: Connects custom data sources to large language models (LLMs).
- **MLflow**: Tracks, logs, and evaluates models.
- **UpTrain**: Provides metrics for evaluating query pipelines, such as context relevance, factual accuracy, and response completeness.

## Features
- Log and query LlamaIndex models using MLflow.
- Evaluate query responses with UpTrain's callback handler.
- Analyze model metrics and performance in MLflow and UpTrain dashboards.

## Setup

### Prerequisites
- Python 3.8 or above
- Git
- Docker (for UpTrain dashboard)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/maheerarshad-emumba/llamaindex-integration.git
   cd llamaindex-integration
2. Create a Virtual Environment and Activate It
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
    ```bash
    pip install -r requirements.txt
4. Add Your OpenAI API Key to a .env File
    OPENAI_API_KEY=your-api-key-here

### Usage
#### MLflow Integration
1. Run the script to log and query models
    ```bash
    python test-mlflow.py

2. Start the MLflow UI
    ```bash
    mlflow ui

#### UpTrain Integration
1. Run the script to evaluate query responses
    ```bash
    python test-uptrain.py

2. Start the UpTrain dashboard
    ```bash
    bash run_uptrain.sh
