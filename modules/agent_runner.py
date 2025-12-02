from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
#from .image_search import image_search_tool
#from .llm import llm_response
#from .memory import add_to_memory

import os
from modules.agent_vto import VTOAgent
from modules.llm import init_llm

def run_agent():
    model_name = os.environ.get("VTO_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # Replace with your model
    token = os.environ.get("VTO_LLM_TOKEN")

    llm = init_llm(model_name=model_name, token=token)
    agent = VTOAgent(llm=llm)

    print("VTO Agent ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Optionally ask for user & garment images
        user_img_path = input("User image path (or press Enter to skip): ").strip() or None
        garment_img_path = input("Garment image path (or press Enter to skip): ").strip() or None

        try:
            response, images = agent.handle_input(user_input, user_img_path, garment_img_path)
        except Exception as e:
            response = f"Error: {e}"
            images = []

        print("\nAgent:", response)
        if images:
            print("\nImages found/generated:")
            for i, img in enumerate(images, 1):
                print(f"{i}. {img['title']}: {img['url']}")

        print("\n--- Memory updated ---\n")

#if __name__ == "__main__":
#    run_agent()
