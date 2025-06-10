class WebQueryHandler:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
    SYSTEM_PROMPT = """You are a helpful AI assistant focused on providing web search results. Follow these guidelines:
    1. Always cite sources for information
    2. Provide relevant and accurate search results
    3. Maintain professional communication
    4. Synthesize information from multiple sources
    5. Indicate when information might be outdated
    6. Verify credibility of sources when possible
    7. Present information in a clear, organized manner
    8. Acknowledge limitations in search results
    9. Protect user privacy and security
    10. Flag any potentially unreliable information"""

    def __init__(self):
        # Initialize Azure OpenAI and DuckDuckGo
        self.azure_model = AzureOpenAIChat(
            id=os.getenv("AZURE_OPENAI_MODEL_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        )
        self.agent = Agent(
            model=self.azure_model,
            tools=[DuckDuckGo()],
            instructions="Always include the sources",
            description="This is the agent for searching content from the web",
            show_tool_calls=True,
            markdown=True,
            debug=True
        )
        self.max_tokens = 800
        self.temperature = 0.7
        self.platform_name = "azure"

    def handle_web_query(self, query: str, user_id: str, chat_id: str, workspace: str, 
                        logger_params: Dict = {}) -> Iterator[Tuple[bool, str]]:
        """
        Handle a web search query and yield streaming responses.
        """
        try:
            now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            today_date = datetime.now().strftime('%Y-%m-%d')
            
            # Initialize database connection
            db = initiate_database('RagWebData', workspace, logger_params)
            chat_meta = {}

            # Initialize or retrieve chat history
            if not chat_id:
                # Initialize new chat
                system_message = {"role": self.SYSTEM, "content": self.SYSTEM_PROMPT}
                chat_meta = {
                    'user_id': user_id,
                    'created_on': now,
                    'chat_conversations': [system_message],
                    'chat_flow': 'web_search',
                    'citations': [[]],
                    'analytics': [{
                        "summary": "",
                        "keywords": [],
                        "sample_questions": []
                    }],
                    'feedback': [{
                        "feedback": None
                    }],
                    'dates': [{
                        "query_date": None,
                        "feedback_date": None
                    }]
                }
            else:
                # Retrieve existing chat
                status, record = db.fetch_one_record({'_id': chat_id}, logger_params=logger_params)
                if not status:
                    yield False, "Failed to retrieve chat history"
                    return

                chat_meta.update(record)

            # Add user message
            chat_meta['chat_conversations'].append({
                "role": self.USER,
                "content": query
            })

            # Perform web search using Agent
            run_response: RunResponse = self.agent.run(query)
            web_search = run_response.content
            web_citations = run_response.sources if hasattr(run_response, 'sources') else []

            # Stream the response
            for chunk in web_search.split():  # Simple streaming simulation
                yield True, json.dumps({"text": chunk + " "})

            # Prepare complete assistant response
            assistant_response = {
                "role": self.ASSISTANT,
                "rag": {
                    "content": "",
                    "citations": []
                },
                "non_rag": {
                    "content": "",
                    "citations": []
                },
                "web": {
                    "content": web_search,
                    "citations": web_citations
                }
            }

            # Update chat metadata
            chat_meta['chat_conversations'].append(assistant_response)
            chat_meta['citations'].append(web_citations)
            chat_meta['analytics'].append({
                "summary": "",
                "keywords": [],
                "sample_questions": []
            })
            chat_meta['feedback'].append({
                "feedback": 2,
                "assessment": None,
                "feedback_description": ""
            })
            chat_meta['dates'].append({
                "query_date": today_date,
                "feedback_date": None
            })

            # Generate title if needed
            if "title" not in chat_meta:
                title_response = self.agent.run("Generate a short, descriptive title for this conversation: " + query)
                chat_meta["title"] = title_response.content

            # Update metadata
            chat_meta['updated_on'] = now

            # Handle AWS Decimal conversion if needed
            if self.platform_name == 'aws':
                chat_meta = json.loads(json.dumps(chat_meta), parse_float=Decimal)

            # Update or insert in database
            if chat_id:
                status, message = db.update_one_record(chat_id, chat_meta, logger_params=logger_params)
            else:
                status, chat_id = db.insert_single_record(chat_meta, logger_params=logger_params)

            if not status:
                yield False, message if chat_id else "Failed to update database"
                return

            # Return final success response
            final_meta = {'_id': ObjectId(chat_id)} if self.platform_name == 'azure' else {'_id': chat_id}
            yield True, json_util.dumps(final_meta)

        except Exception as e:
            print(f"Error in web search processing: {str(e)}")
            yield False, f"Failed to process web search query: {str(e)}"

    def extract_citations(self, run_response: RunResponse) -> list:
        """Extract citations from the run response."""
        try:
            return run_response.sources if hasattr(run_response, 'sources') else []
        except Exception as e:
            print(f"Error extracting citations: {str(e)}")
            return []