import os
import json
import torch
from PIL import Image
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, AgentType, initialize_agent
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ddgs import DDGS
from vto_integration import VTOAgentModule

MEMORY_FILE = "vto_memory.json"

# ===== MEMORY =====
def load_memory_data():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_memory_data(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_to_memory(user_input, agent_response, images):
    memory = load_memory_data()
    memory.setdefault("conversations", []).append({
        "user_input": user_input,
        "agent_response": agent_response,
        "images": images
    })
    save_memory_data(memory)

# ===== SEARCH TOOL =====
def search_images(query: str, max_results: int = 5):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.images(query):
                results.append({"title": r["title"], "url": r["image"]})
                if len(results) >= max_results:
                    break
    except:
        pass
    return results

image_search_tool = Tool(
    name="Image Search",
    func=search_images,
    description="Search images on DuckDuckGo and return top 5 results"
)

# ===== LLM =====
def build_llm(model_name, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto", 
            torch_dtype=torch.float16, 
            token=token
        )
    except:
        # Fallback to CPU if GPU is somehow full (unlikely with lazy loading)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            token=token
        )
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=200, 
        do_sample=True, 
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=pipe)

# ===== AGENT =====
def build_agent(llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[image_search_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    return agent

# ===== MAIN CLASS =====
class VTOAgent:
    def __init__(self, llm, lora_path: str):
        self.llm = llm
        self.agent = build_agent(llm)
        # Initialize VTO module (does NOT load heavy model yet)
        self.vto_module = VTOAgentModule(lora_path=lora_path)

    def handle_input(self, user_input: str, user_img_path: str = None, garment_img_path: str = None):
        images = search_images(user_input)
        agent_response = ""

        # --- Try-On Logic ---
        if user_img_path and garment_img_path:
            try:
                # Use the module to manage memory (Load -> Process -> Unload)
                inpainted_img = self.vto_module.process_tryon(user_img_path, garment_img_path)
                
                # Save output
                output_path = "outputs/final_tryon.png"
                inpainted_img.save(output_path)
                
                agent_response = f"VTO Success! Image saved at {output_path}"
                images.insert(0, {"title": "VTO Result", "url": output_path})
            except Exception as e:
                import traceback
                traceback.print_exc()
                agent_response = f"VTO Error: {e}"
        
        # --- Chat Logic ---
        else:
            try:
                agent_response = self.agent.run(user_input)
            except Exception as e:
                agent_response = f"Agent Error: {e}"
        # Save to memory
        add_to_memory(user_input, agent_response, images)
        return agent_response, images
