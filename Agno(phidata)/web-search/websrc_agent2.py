import os
from typing import Iterator
from phi.agent import Agent, RunResponse
from phi.model.azure import AzureOpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api=os.getenv("Phi_API_Key")
azure_model = AzureOpenAIChat(
    id=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)


agent = Agent(
    name ="Web Search Agent",
    model=azure_model,
    tools  = [DuckDuckGo()],
    instructions = "Always include the sources",
    description ="This is the agent for searching content from the web",
    show_tool_calls = True,
    markdown = True,
    debug = True
)

# Get the response in a variable
# run: RunResponse = agent.run("Share a 2 sentence horror story.")
# print(run.content)

# Print the response on the terminal
#agent.print_response("About AIXponent.",stream=True)
app = Playground(agents= ["Web Search Agent"]).get_app()

if __name__ == "__main__":
    serve_playground_app("websrc_agent2:app", reload=True)