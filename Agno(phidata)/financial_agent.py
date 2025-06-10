from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

# Web Search agent
web_search_agent = Agent(
    name="Web search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools =[DuckDuckGo()],
    instructions="Always include the scources in the search results",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

# Fincial Agent
finance_agent = Agent(
    name="Financial AI Agent",
    role="Your task is to find the finance information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True, company_info=True, historical_prices=True, technical_indicators=True, income_statements=True)],
    instructions="Use tables to display the data",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

multi_ai_agent = Agent(
    team=(web_search_agent, finance_agent),
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include the scources in the search results","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

multi_ai_agent.print_response("Summerize the analyst recommendation and share the latest news from Nvidia",stream=True)