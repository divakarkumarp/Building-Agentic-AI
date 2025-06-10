from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Set up your API key (you'll need to get one from OpenAI)
os.environ["OPENAI_API_KEY"] = "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"

# Define the Agent
research_agent = Agent(
    role='Research Analyst',
    goal='Conduct thorough research and provide detailed analysis',
    backstory="""You're an experienced research analyst with a knack for uncovering 
    valuable insights from complex data. You excel at breaking down information 
    and presenting it in a clear, concise manner.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
)

# Define a Task for the Agent
research_task = Task(
    description="""Research the current trends in artificial intelligence development 
    and provide a summary of key findings. Include at least 3 specific examples 
    of recent advancements.""",
    expected_output="A detailed summary of AI trends with 3+ specific examples",
    agent=research_agent
)

# Create a Crew with the Agent and Task
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True,  # Higher verbosity for more detailed logs
)

# Execute the Crew
if __name__ == "__main__":
    result = crew.kickoff()
    print("\nResults:")
    print(result)