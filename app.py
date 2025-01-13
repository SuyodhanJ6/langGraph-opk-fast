from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import os

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Set Tavily API key in environment variable
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY  # Tavily requires the key to be set in env vars
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import asyncio
from datetime import datetime
import uuid
import logging
from contextlib import asynccontextmanager
import uvicorn
from opik.integrations.langchain import OpikTracer

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from typing_extensions import TypedDict
from typing import Annotated
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import redis
from redis.client import Redis
import pickle
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID for the conversation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")

# Global state management
class GlobalState:
    def __init__(self):
        self.active_sessions: Dict[str, ConversationBufferWindowMemory] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.graph = None
        self.llm = None
        self.opik_tracer = None  # Opik tracer instance

state = GlobalState()

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant with access to search capabilities. Follow these guidelines:
1. Maintain context awareness
2. Use search tools when needed
3. Provide clear, organized responses
4. Be transparent about limitations
5. Maintain professional yet conversational tone
6. Solve problems systematically"""

# State definition for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    memory: ConversationBufferWindowMemory

async def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Get or create session memory."""
    if session_id not in state.active_sessions:
        state.active_sessions[session_id] = ConversationBufferWindowMemory(
            k=2,
            return_messages=True,
            memory_key="chat_history"
        )
        state.session_locks[session_id] = asyncio.Lock()
    return state.active_sessions[session_id]

def create_graph():
    """Create and configure the conversation graph."""
    graph_builder = StateGraph(State)
    
    # Initialize tools and LLM with API keys
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY  # Set environment variable for Tavily
    search_tool = TavilySearchResults(max_results=2)
    llm = ChatGroq(
        temperature=0.1,
        model_name="mixtral-8x7b-32768",
        streaming=True,
        api_key=GROQ_API_KEY
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Configure LLM with tools
    llm_with_tools = llm.bind(
        prompt=prompt,
        config={"metadata": {"system_prompt": SYSTEM_PROMPT}}
    ).bind_tools([search_tool])

    # Define chatbot node
    def chatbot(state: State):
        current_messages = state["messages"]
        memory = state["memory"]
        
        if current_messages:
            last_message = current_messages[-1]
            if hasattr(last_message, "content"):
                if last_message.type == "human":
                    memory.save_context(
                        {"input": last_message.content},
                        {"output": ""}
                    )
                elif last_message.type == "ai" and len(current_messages) > 1:
                    memory.save_context(
                        {"input": current_messages[-2].content},
                        {"output": last_message.content}
                    )

        chat_history = memory.load_memory_variables({})["chat_history"]
        formatted_messages = []
        
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        if current_messages and hasattr(current_messages[-1], "content"):
            formatted_messages.append({"role": "user", "content": current_messages[-1].content})

        response = llm_with_tools.invoke(formatted_messages)
        return {"messages": [response]}

    # Add nodes and edges
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=[search_tool])
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the FastAPI application."""
    try:
        # Initialize graph on startup
        state.graph = create_graph()
        # Initialize Opik tracer
        state.opik_tracer = OpikTracer(graph=state.graph.get_graph(xray=True))
        logger.info("Conversation graph and Opik tracer initialized successfully")
        yield
    finally:
        # Cleanup on shutdown
        if state.opik_tracer:
            state.opik_tracer.flush()
        state.active_sessions.clear()
        state.session_locks.clear()
        logger.info("Application shutdown complete")

app = FastAPI(
    title="AI Chat Service",
    description="Production-ready AI chat service with session management",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_chat_request(
    request: ChatRequest,
    memory: ConversationBufferWindowMemory,
    session_id: str
) -> str:
    """Process a chat request using the conversation graph."""
    try:
        # Configure stream with Opik tracer
        events = state.graph.stream(
            {
                "messages": [("user", request.message)],
                "memory": memory,
                "metadata": {
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_prompt": SYSTEM_PROMPT,
                    "model": "mixtral-8x7b-32768",
                    "session_type": "interactive",
                    **request.metadata
                }
            },
            stream_mode="values",
            config={"callbacks": [state.opik_tracer]} if state.opik_tracer else None
        )

        last_response = None
        for event in events:
            if "messages" in event:
                message = event["messages"][-1]
                if hasattr(message, "content"):
                    last_response = message.content
                elif hasattr(message, "tool_calls"):
                    # Handle tool calls if needed
                    logger.info(f"Tool call executed: {message.tool_calls}")

        if not last_response:
            raise ValueError("No response generated")
        
        # Ensure all traces are logged
        if state.opik_tracer:
            state.opik_tracer.flush()
        
        return last_response

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing chat request")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Handle chat requests with session management and concurrent request handling."""
    try:
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create memory for this session
        try:
            memory = await get_session_memory(session_id)
        except Exception as e:
            logger.error(f"Error getting session memory: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Error initializing chat session"
            )

        # Process request with session lock
        try:
            async with state.session_locks[session_id]:
                response_text = await process_chat_request(request, memory, session_id)
        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Error processing your message"
            )

        # Schedule cleanup in background
        background_tasks.add_task(cleanup_old_sessions)
        
        # Prepare and return response
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "request_metadata": request.metadata,
                "session_info": {
                    "is_new_session": session_id not in state.active_sessions,
                    "session_id": session_id
                }
            }
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )

async def cleanup_old_sessions():
    """Clean up inactive sessions periodically."""
    # Implement session cleanup logic based on your requirements
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
