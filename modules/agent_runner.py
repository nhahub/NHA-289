from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from .image_search import image_search_tool
from .llm import llm_response
from .memory import add_to_memory

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

def run_agent(llm):
    agent = build_agent(llm)
    print("VTO Agent ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        images = image_search_tool.func(user_input)
        try:
            agent_response = agent.run(user_input)
        except Exception as e:
            agent_response = f"Error: {e}"

        print("\nAgent:", agent_response)
        if images:
            print("\nImages found:")
            for i, img in enumerate(images, 1):
                print(f"{i}. {img['title']}: {img['url']}")

        add_to_memory(user_input, agent_response, images)
        print("\nMemory updated!\n")
