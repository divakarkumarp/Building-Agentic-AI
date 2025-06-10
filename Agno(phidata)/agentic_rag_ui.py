from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.knowledge.website import WebsiteKnowledgeBase
import os
from phi.playground import Playground, serve_playground_app


from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# create vectordb
vector_db = Qdrant(
    collection="phidata-qdrant-ytipynb",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    embedder = OllamaEmbedder()
)

URL = "https://www.morganstanley.com/content/dam/msdotcom/en/assets/pdfs/Morgan_Stanley_2023_ESG_Report.pdf"

knowledge_base = PDFUrlKnowledgeBase(
    urls = [URL],
    vector_db = vector_db,
)

agent_ollama = Agent(
    name="Agentic RAG",
    agent_id = "agentic-rag",
    
    model=Ollama(id="llama3.2:latest"),

    knowledge = knowledge_base,
    
    # Add a tool to search the knowledge base which enables agentic RAG.
    # This is enabled by default when `knowledge` is provided to the Agent.
    search_knowledge=True,
    
    # Add a tool to read chat history (default tool)
    read_chat_history=True,
    #add_history_to_messages = True, 
    
    show_tool_calls=True,
    
    # Store the agent sessions
    storage=SqlAgentStorage(table_name="phidata", db_file="agents_rag.db"),
    
    instructions=[
        "Always search your knowledge base first and use it if available.",
        "Share the page number or source URL of the information you used in your response.",
        "If health benefits are mentioned, include them in the response.",
        "Important: Use tables where possible.",
    ],
    
    markdown=True,
)

app = Playground(agents=[agent_ollama]).get_app()

if __name__ == "__main__":
    # Load the knowledge base: Comment after first run as the knowledge base is already loaded
    #knowledge_base.load(upsert=True)
    serve_playground_app("agentic_rag_ui:app", reload=True)