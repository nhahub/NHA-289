import os
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def init_llm(model_name: str, token: str = None, device: str = None, dtype: str = "float16"):
    """
    Initialize LLM with optional device and dtype.
    """
    token = token or os.environ.get("VTO_LLM_TOKEN")
    if token is None:
        raise ValueError("Missing API token. Set environment variable: VTO_LLM_TOKEN")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch_dtype,
        use_auth_token=token
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        device=0 if device == "cuda" else -1
    )

    return HuggingFacePipeline(pipeline=pipe)
