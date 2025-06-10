from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

web_search_agent = Agent(
    name = "Web Search Agent",
    description = "This is the agent for searching content from the web",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools  = [DuckDuckGo()],
    instructions = "Always include the sources",
    show_tool_calls = True,
    markdown = True,
    #debug = True
)

web_search_agent.print_response("About Mahakumbh?", stream=True)