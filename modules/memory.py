import os, json

MEMORY_FILE = "vto_memory.json"

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
