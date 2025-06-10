from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import os
import phi

load_dotenv()
phi.api=os.getenv("Phi_API_Key")

# Web Search agent
web_search_agent = Agent(
    name ="Web Search Agent",
    description ="This is the agent for searching content from the web",
    model =Groq(id="mixtral-8x7b-32768"),
    tools  =[DuckDuckGo()],
    instructions =["Always include the sources"],
    show_tool_calls =True,
    markdown =True,
    debug_mode=True
)


# Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    description ="Your task is to find the finance information",
    model =Groq(id="llama-3.3-70b-specdec"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode = True
)

app = Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)