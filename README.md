# Virtual Try-On Tool

This project implements a Virtual Try-On (VTO) agent using a HuggingFace LLM and LangChain, allowing users to chat with the agent and perform try-on tasks with garment and user images.

## Setup Instructions

#### 1. Clone the repository
```
git clone https://github.com/nhahub/NHA-289
cd NHA-289
```

#### 2. Install Dependencies
```
pip install -r requirements.txt
```

#### 3. Set Environment Variables

Before running the agent, set the environment variables:

```
import os

os.environ["VTO_LLM_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"
os.environ["VTO_LLM_TOKEN"] = "your_huggingface_token"   # put your hugging face access token here
os.environ["VTO_LORA_PATH"] = "ckpts/lora.safetensors"   
os.environ["VTO_OUTPUT_DIR"] = "outputs"     
```

After setting them, run:

```
!python agent_runner.py
```
