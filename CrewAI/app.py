import os
from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Create a DuckDuckGo search tool compatible with CrewAI
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="DuckDuckGo Search",
    func=lambda query: search.run(query),  # Ensure it accepts a string
    description="Search the web using DuckDuckGo for up-to-date information. Provide a string query."
)

# Define agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and data science",
    backstory="""You are an expert at a technology research group, 
    skilled in identifying trends and analyzing complex data.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7),
    tools=[search_tool]
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a content strategist known for 
    making complex tech topics interesting and easy to understand.""",
    verbose=True,
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7),
    allow_delegation=True
)

# Define tasks
task1 = Task(
    description="""Conduct a detailed analysis of AI advancements in 2024. 
    Identify major trends, emerging technologies, and their impacts across industries. 
    Use the DuckDuckGo search tool with queries like 'AI advancements 2024 trends technologies impacts' 
    to gather current data and compile a detailed report.""",
    agent=researcher,
    expected_output="A comprehensive report on 2024's AI advancements, including major trends, technologies, and their impacts."
)

task2 = Task(
    description="""Using the researcher's report, write an engaging blog post about major AI advancements in 2024. 
    Ensure it’s clear, captivating, and tailored for tech enthusiasts. 
    The post should be at least 4 paragraphs long. Include trends like Generative AI, AI Ethics, 
    AI in Healthcare, Edge Computing, NLP, Sustainability, Augmented Intelligence, and AI in Finance.""",
    agent=writer,
    expected_output="A blog post of at least 4 paragraphs, engagingly covering 2024's major AI advancements for tech enthusiasts."
)

# Instantiate crew with sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    process="sequential"
)

# Execute the crew
if __name__ == "__main__":
    print("Crew starting work...")
    result = crew.kickoff()
    print("######################")
    print("Final Result:")
    print(result)