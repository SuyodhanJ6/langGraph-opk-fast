from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from opik.integrations.langchain import OpikTracer
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
import opik
import os

# Configure Opik for local server
os.environ["OPIK_SERVER_URL"] = "http://71.138.22.131:5173"
os.environ["OPIK_LOCAL"] = "true"
opik.configure(use_local=True)

# Define the system prompt
SYSTEM_PROMPT = """You are a helpful AI assistant with access to search capabilities. Follow these guidelines:

1. CONTEXT AWARENESS:
   - Maintain context from the conversation history
   - Reference previous interactions when relevant
   - Ask for clarification if the user's request is ambiguous

2. SEARCH TOOL USAGE:
   - Use the search tool when you need current or factual information
   - Cite sources when providing information from searches
   - Verify information before presenting it to the user

3. RESPONSE STRUCTURE:
   - Provide clear, organized responses
   - Break down complex information into digestible parts
   - Use appropriate formatting for better readability

4. LIMITATIONS:
   - Be transparent about what you can and cannot do
   - Acknowledge when you need to search for information
   - Admit if you're unsure about something

5. INTERACTION STYLE:
   - Be professional yet conversational
   - Show empathy and understanding
   - Adapt your tone to match the user's needs

6. PROBLEM SOLVING:
   - Think through problems step by step
   - Explain your reasoning process
   - Offer multiple solutions when appropriate

Remember to always prioritize accuracy and helpfulness in your responses."""

# Define state with memory
class State(TypedDict):
    messages: Annotated[list, add_messages]
    memory: ConversationBufferWindowMemory

# Initialize memory with k=2 to keep last 2 exchanges
memory = ConversationBufferWindowMemory(
    k=2,
    return_messages=True,
    memory_key="chat_history"
)

# Create graph builder with our state
graph_builder = StateGraph(State)

# Create search tool and LLM
search_tool = TavilySearchResults(max_results=2)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="messages"),
])

# Initialize the LLM
llm = ChatGroq(
    temperature=0.1,
    model_name="mixtral-8x7b-32768",
    streaming=True  # Enable streaming for better tracing
)

# Initialize LLM chain with prompt and tools
llm_with_tools = llm.bind(
    prompt=prompt,
    config={
        "metadata": {
            "system_prompt": SYSTEM_PROMPT,
            "run_name": "agent_execution"
        }
    }
).bind_tools([search_tool])

def chatbot(state: State):
    # Get current messages
    current_messages = state["messages"]
    
    # If there are new messages, update memory
    if current_messages:
        last_message = current_messages[-1]
        if hasattr(last_message, "content"):
            # Save to memory if it's a regular message
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

    # Get recent history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # Transform messages into the correct format
    formatted_messages = []
    
    # Add chat history
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
    
    # Add the current message if it exists
    if current_messages and hasattr(current_messages[-1], "content"):
        formatted_messages.append({"role": "user", "content": current_messages[-1].content})

    # Get response using properly formatted messages
    response = llm_with_tools.invoke(formatted_messages)
    return {"messages": [response]}

# Add nodes to graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[search_tool])
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()

def run_agent(user_input: str, debug: bool = True):
    # Create OpikTracer instance with basic configuration
    opik_tracer = OpikTracer(graph=graph.get_graph(xray=True))
    
    # Stream with OpikTracer callback
    events = graph.stream(
        {
            "messages": [("user", user_input)],
            "metadata": {
                "system_prompt": SYSTEM_PROMPT,
                "model": "mixtral-8x7b-32768",
                "session_type": "interactive",
                "user_input": user_input
            }
        },
        stream_mode="values",
        config={"callbacks": [opik_tracer]}
    )
    
    if debug:
        print("\n=== Memory Status ===")
        print("Current memory contents:")
        chat_history = memory.load_memory_variables({})["chat_history"]
        for i, msg in enumerate(chat_history, 1):
            print(f"{i}. {msg.type}: {msg.content[:100]}...")
    
    last_response = None
    for event in events:
        if "messages" in event:
            message = event["messages"][-1]
            if hasattr(message, "content"):
                last_response = message.content
                print(f"\nAssistant: {message.content}")
            else:
                print(f"\nAssistant used tool: {message.tool_calls[0]['name']}")
                if debug:
                    print(f"Tool args: {message.tool_calls[0]['args']}")
    
    # Ensure all traces are logged
    opik_tracer.flush()
    return last_response

if __name__ == "__main__":
    print("Chat with the agent (type 'exit' to quit)")
    print(f"Using Opik server at: {os.getenv('OPIK_SERVER_URL')}")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        run_agent(user_input, debug=True)
# from typing import Annotated
# from typing_extensions import TypedDict
# from langchain_groq import ChatGroq
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.schema import HumanMessage, AIMessage
# from opik.integrations.langchain import OpikTracer
# import opik
# import os

# # Configure Opik for local server - only one configuration method
# os.environ["OPIK_SERVER_URL"] = "http://71.138.22.131:5173"
# os.environ["OPIK_LOCAL"] = "true"
# opik.configure(use_local=True)

# # Define state with memory
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     memory: ConversationBufferWindowMemory

# # Initialize memory with k=2 to keep last 2 exchanges
# memory = ConversationBufferWindowMemory(
#     k=2,
#     return_messages=True,
#     memory_key="chat_history"
# )

# # Create graph builder with our state
# graph_builder = StateGraph(State)

# # Create search tool and LLM
# search_tool = TavilySearchResults(max_results=2)
# llm = ChatGroq(
#     temperature=0.1,
#     model_name="mixtral-8x7b-32768",
# )

# # Bind the tool to the LLM
# llm_with_tools = llm.bind_tools([search_tool])

# def chatbot(state: State):
#     # Get current messages
#     current_messages = state["messages"]
    
#     # If there are new messages, update memory
#     if current_messages:
#         last_message = current_messages[-1]
#         if hasattr(last_message, "content"):
#             # Save to memory if it's a regular message
#             if last_message.type == "human":
#                 memory.save_context(
#                     {"input": last_message.content}, 
#                     {"output": ""}
#                 )
#             elif last_message.type == "ai" and len(current_messages) > 1:
#                 memory.save_context(
#                     {"input": current_messages[-2].content},
#                     {"output": last_message.content}
#                 )

#     # Get recent history from memory
#     chat_history = memory.load_memory_variables({})["chat_history"]
    
#     # Create new context with only recent history
#     limited_context = []
#     for msg in chat_history:
#         if isinstance(msg, HumanMessage):
#             limited_context.append(("user", msg.content))
#         elif isinstance(msg, AIMessage):
#             limited_context.append(("assistant", msg.content))
    
#     # Add the current message if it exists
#     if current_messages and hasattr(current_messages[-1], "content"):
#         limited_context.append(
#             ("user", current_messages[-1].content)
#         )

#     # Get response using limited context
#     response = llm_with_tools.invoke(limited_context)
#     return {"messages": [response]}

# # Add nodes to graph
# graph_builder.add_node("chatbot", chatbot)
# tool_node = ToolNode(tools=[search_tool])
# graph_builder.add_node("tools", tool_node)

# # Add edges
# graph_builder.add_conditional_edges(
#     "chatbot",
#     tools_condition,
# )
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")

# # Compile the graph
# graph = graph_builder.compile()

# def run_agent(user_input: str, debug: bool = True):
#     # Create OpikTracer instance - simplified configuration
#     opik_tracer = OpikTracer(graph=graph.get_graph(xray=True))
    
#     # Stream with OpikTracer callback
#     events = graph.stream(
#         {"messages": [("user", user_input)]},
#         stream_mode="values",
#         config={"callbacks": [opik_tracer]}
#     )
    
#     if debug:
#         print("\n=== Memory Status ===")
#         print("Current memory contents:")
#         chat_history = memory.load_memory_variables({})["chat_history"]
#         for i, msg in enumerate(chat_history, 1):
#             print(f"{i}. {msg.type}: {msg.content[:100]}...")
    
#     last_response = None
#     for event in events:
#         if "messages" in event:
#             message = event["messages"][-1]
#             if hasattr(message, "content"):
#                 last_response = message.content
#                 print(f"\nAssistant: {message.content}")
#             else:
#                 print(f"\nAssistant used tool: {message.tool_calls[0]['name']}")
#                 if debug:
#                     print(f"Tool args: {message.tool_calls[0]['args']}")
    
#     # Ensure all traces are logged
#     opik_tracer.flush()
#     return last_response

# if __name__ == "__main__":
#     print("Chat with the agent (type 'exit' to quit)")
#     print(f"Using Opik server at: {os.getenv('OPIK_SERVER_URL')}")
    
#     while True:
#         user_input = input("\nUser: ")
#         if user_input.lower() in ["exit", "quit", "q"]:
#             break
            
#         run_agent(user_input, debug=True)
