import sys
from crewai import Agent, Task
import os
from dotenv import load_dotenv
from crewai import Crew, Process
from langchain_openai import AzureChatOpenAI

load_dotenv()

default_llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2024-08-01-preview"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://aoi-gptxponent-sc.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"),
    api_key=os.environ.get("AZURE_OPENAI_KEY")
)


# Create a researcher agent
researcher = Agent(
  role='Senior Researcher',
  goal='Discover groundbreaking technologies',
  verbose=True,
  llm=default_llm,
  backstory='A curious mind fascinated by cutting-edge innovation and the potential to change the world, you know everything about tech.'
)

# Task for the researcher
research_task = Task(
  description='Identify the next big trend in AI',
  expected_output='5 paragraphs on the next big AI trend',
  agent=researcher  # Assigning the task to the researcher
)


# Instantiate your crew
tech_crew = Crew(
  agents=[researcher],
  tasks=[research_task],
  process=Process.sequential  # Tasks will be executed one after the other
)

# Begin the task execution
tech_crew.kickoff()