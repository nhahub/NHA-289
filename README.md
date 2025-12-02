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

## Folder Structure

```
NHA-289/
│
├── assets/                      # All images used in the project
│   ├── clothes/                 # Garment images
│   │   ├── Black-Full-T-Shirt.png
│   │   ├── blue-t-shirt.jpeg
│   │   ├── clo.png
│   │   ├── cloth_test_2.webp
│   │   ├── final_cloth.jpg
│   │   ├── jacket_jeans_remv.png
│   │   ├── leather_pants.png
│   │   ├── pants.webp
│   │   └── sweater.webp
│   │
│   └── user_images/             # User uploaded images
│       ├── .gitkeep
│       ├── final_test.jpg
│       ├── lady_2.jpg
│       ├── t-shirt-try.avif
│       └── test_woman.webp
│
├── ckpts/                       # fine tuned LoRA weights
│   └── pytorch_lora_weights.safetensors
│
├── modules/                     # All project modules
│   ├── agent_runner.py
│   ├── agent_vto.py
│   ├── image_search.py
│   ├── inpainting.py
│   ├── llm.py
│   ├── memory.py
│   ├── segmentation.py
│   └── vto_integration.py
│
├── outputs/                     # Virtual try-on results saved here
│
├── README.md
├── final_project_test.ipynb
└── requirements.txt

```
