from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Set up your API key
os.environ["OPENAI_API_KEY"] = "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"


# Get user query
user_input = input("Enter your ESG-related question or topic: ")

# Define Agents
esg_specialist = Agent(
    role="ESG Specialist",
    goal="Provide deep domain knowledge on Environmental, Social, and Governance topics",
    backstory="""You're an ESG subject matter expert with in-depth knowledge 
    of regulatory requirements, global ESG standards, and frameworks such as GRI, SASB, and TCFD.""",
    instructions="Always include the scources in the search results",
    allow_delegation=False,
    verbose=True,
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5),
)

esg_advisor = Agent(
    role="ESG Advisor",
    goal="Advise organizations on how to integrate ESG into their strategy and operations",
    backstory="""You're a strategic advisor experienced in guiding businesses to 
    adopt ESG principles, improve stakeholder relationships, and create sustainable value.""",
    instructions="Always include the scources in the search results",
    allow_delegation=False,
    verbose=True,
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5),
)

esg_performance_analyst = Agent(
    role="ESG Performance Analyst",
    goal="Analyze ESG performance metrics and suggest areas for improvement",
    backstory="""You're an analyst skilled in evaluating ESG KPIs, sustainability reporting, 
    and benchmarking performance against industry peers.""",
    instructions="Always include the scources in the search results",
    allow_delegation=False,
    verbose=True,
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5),
)

# Define Tasks
task_specialist = Task(
    description=f"Break down the ESG topic: '{user_input}'. Explain its core concepts, standards involved, and implications.",
    expected_output="An informative explanation including standards like GRI, SASB, TCFD.",
    agent=esg_specialist,
)

task_advisor = Task(
    description=f"Advise how a company can implement ESG principles related to: '{user_input}'. Include benefits and strategic suggestions.",
    expected_output="Actionable advice for ESG integration into business strategy.",
    agent=esg_advisor,
)

task_analyst = Task(
    description=f"Evaluate ESG performance indicators for '{user_input}' and suggest how to track and improve them.",
    expected_output="Detailed performance metrics and improvement strategies.",
    agent=esg_performance_analyst,
)

# Create Crew
crew = Crew(
    agents=[esg_specialist, esg_advisor, esg_performance_analyst],
    tasks=[task_specialist, task_advisor, task_analyst],
    verbose=True,
)

# Run the Crew
if __name__ == "__main__":
    result = crew.kickoff()
    print("\nFinal ESG Report:")
    print(result)
