import os
from typing import Iterator
from phi.agent import Agent, RunResponse
from phi.model.azure import AzureOpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

class AgentQueryHandler:
    SYSTEM_PROMPT = """You are a helpful AI assistant that searches the web for accurate information. 
    Always include sources in your responses and maintain:
    1. Factual, well-researched information
    2. Professional communication
    3. Clear attribution of sources
    4. Educational and constructive responses"""

    def __init__(self):
        load_dotenv()
        
        # Initialize Azure OpenAI model
        self.azure_model = AzureOpenAIChat(
            id=os.getenv("AZURE_OPENAI_MODEL_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        )
        
        # Initialize agent with tools
        self.agent = Agent(
            model=self.azure_model,
            tools=[DuckDuckGo()],
            instructions="Always include the sources",
            description="This is the agent for searching content from the web",
            show_tool_calls=True,
            markdown=True,
            debug=True
        )

    def handle_query(self, query: str, stream: bool = True) -> Iterator[str]:
        """Handle a query and yield streaming responses"""
        try:
            # Get streaming response from agent
            run_response = self.agent.run(query, stream=stream)
            
            if stream:
                # Stream the response
                for chunk in run_response:
                    if isinstance(chunk, str):
                        yield chunk
            else:
                # Return complete response
                yield run_response.content

        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            yield f"Failed to process query: {str(e)}"

# Usage example:
if __name__ == "__main__":
    handler = AgentQueryHandler()
    
    # Example with streaming
    query = "about clifornia fire?"
    for response_chunk in handler.handle_query(query):
        print(response_chunk, end='', flush=True)
    
    # Example without streaming
    response = next(handler.handle_query(query, stream=False))
    print(f"\n\nComplete response: {response}")