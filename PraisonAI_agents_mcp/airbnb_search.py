from praisonaiagents import Agent, MCP

search_agent = Agent(
    instructions="""You help book apartments on Airbnb.""",
    llm="gpt-4o-mini",
    tools=MCP("npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt")
)

search_agent.start("I want to book an apartment in Delhi for 2 nights. 04/28 - 04/30 for 2 adults and location should be near Connaught Place.")
    