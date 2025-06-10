from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

import os
load_dotenv()

Topic = "Medical Industry using Generative AI"

# TOOL 1
default_llm = LLM(model="gpt-4-mini")
#SERPER_API_KEY=os.getenv("SERPER_API_KEY")
# Tool 2 
search_tool = SerperDevTool(n=10)

# Agent 1 - Research Analyst
research_analyst = Agent(
    role="Research Analyst",
    goal=f"Research, analyze, and synthesize the best articles on the {Topic} from reliable sources.",
    backstory=("""
        You are a researcher specializing in analyzing complex topics.
        You excel at finding, analyzing, and synthesizing information from various sources.
        Your goal is to deliver a structured research report with credible insights and citations.
        You are detail-oriented, a critical thinker, and can work independently to meet deadlines.
    """),
    verbose=True,
    allow_delegation=False,
    llm=default_llm,
    tools=[search_tool],
)

# Agent 2 - Content Writer
content_writer = Agent(
    role="Content Writer",
    goal=f"Write a comprehensive and easy-to-read report on {Topic} based on research findings.",
    backstory=("""
        You are a skilled content writer specializing in transforming research into clear, engaging reports.
        You excel at organizing complex information into well-structured documents.
        Your goal is to deliver a polished report with clear sections and verified data.
    """),
    verbose=True,
    allow_delegation=False,
    llm=default_llm,
    tools=[search_tool],
)

# Task 1 - Research Task
research_task = Task(
    description=f"""
        1. Conduct comprehensive research on {Topic}, including:
            - Finding the best articles from reliable sources.
            - Analyzing the information and extracting key insights.
            - Collecting statistical data and market trends.
        2. Evaluate source credibility and fact-check information.
        3. Organize findings into a structured research brief with citations.
    """,
    expected_output="""A detailed research report including:
        - Executive summary of key findings.
        - Comprehensive analysis of current trends and developments.
        - Verified facts, statistics, and market insights.
        - Clear categorization of main themes and patterns.
        - Properly formatted with bullet points for easy reference.
    """,
    agent=research_analyst,
)

# Task 2 - Content Writing Task
writing_task = Task(
    description=f"""
        1. Write a clear, structured report on {Topic} based on the research findings.
        2. Organize the report into easy-to-follow sections with proper headings.
        3. Ensure the report is well-written and easy to understand.
        4. Include citations and references for all claims.
    """,
    expected_output="""A professional report including:
        - Executive summary of the key findings.
        - In-depth analysis of major trends and developments.
        - Clear, easy-to-read structure with bullet points.
        - Verified citations and links to original sources.
        - Well-formatted sections for easy navigation.
    """,
    agent=content_writer,
)

# Crew Setup
crew = Crew(
    agents=[research_analyst, content_writer],
    tasks=[research_task, writing_task],
    verbose=True
)

# Execute Workflow
result = crew.kickoff(inputs= {"topic": Topic})
print(result)
