from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from tools.crop_tool import crop_tool
import os

# 🔑 Set your API key
os.environ["OPENAI_API_KEY"] = "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"

# LLM
llm = ChatOpenAI(temperature=0)

# Memory
memory = ConversationBufferMemory()

# Tools
tools = [
    Tool(
        name="Crop Recommendation Tool",
        func=crop_tool,
        description="Input format: N,P,K,temperature,humidity,rainfall"
    )
]

# Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True
)

# Function to call from UI
def run_agent(user_input):
    prompt = f"""
    Analyze the agricultural data and:
    1. Recommend best crop
    2. Give reason
    3. Suggest fertilizer

    Input: {user_input}
    """
    return agent.run(prompt)


# CLI test
if __name__ == "__main__":
    while True:
        query = input("Enter values (N,P,K,temp,humidity,rainfall): ")
        print(run_agent(query))