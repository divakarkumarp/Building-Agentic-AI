from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.knowledge.website import WebsiteKnowledgeBase
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

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

# Load the knowledge base: Comment after first run as the knowledge base is already loaded
knowledge_base.load(upsert=True)



agent_ollama = Agent(
    model=Ollama(id="llama3.2:latest"),
    knowledge=knowledge_base,
    # Set as False because Agents default to `search_knowledge=True`
    search_knowledge=False,
    show_tool_calls=True,
    markdown=True,
    
    #storage=SqlAgentStorage(table_name="phidata", db_file="agents_rag.db"),
    add_history_to_messages=True,
)

agent_ollama.print_response(
  "what is goal of this 2023 ESG Report document", stream=True
)