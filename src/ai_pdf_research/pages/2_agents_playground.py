import streamlit as st
from langchain.agents import Tool, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.tools import Tool
from langchain_community.tools.calculator.tool import Calculator



from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

st.title("🛠 Agent Playground")

api_key = os.getenv("GOOGLE_API_KEY")
model = os.getenv("MODEL")
llm = ChatGoogleGenerativeAI(
    model=model,
    google_api_key=api_key
)

calc= Calculator()
calc_tool = Tool.from_function(
    func=calc.run,
    name="Calculator",
    description="Useful for when you need to do math"
)


agent_executor = initialize_agent(
    tools=[calc_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent_executor.run("What is 45 times 78?")
st.write("Response from agent:", response)




# Code for tool setup and agent testing
