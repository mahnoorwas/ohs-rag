from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import GoogleGenerativeAI
from rag_pipeline import query
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Gemini LLM
llm = GoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Tool definition for OHS Retrieval
tools = [
    Tool(
        name="OHS Retrieval Augmented Generation",
        func=query,
        description="Useful for answering questions from OHS guidelines and manuals. "
                    "Input should be a safety-related question, and output will be the relevant answer from OHS documents.",
    )
]

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

if __name__ == "__main__":
    print("🦺 OHS Safety Assistant (type 'exit' to quit)")
    while True:
        user_query = input("Enter your OHS question: ")
        if user_query.lower() == "exit":
            break
        result = agent.run(user_query)
        print(result)
