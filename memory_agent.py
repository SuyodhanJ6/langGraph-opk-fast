

from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

# Define state with memory
class State(TypedDict):
    messages: Annotated[list, add_messages]
    memory: ConversationBufferWindowMemory

# Initialize memory with k=2 to keep last 2 exchanges
memory = ConversationBufferWindowMemory(
    k=2,  # Only keep last 2 exchanges
    return_messages=True,
    memory_key="chat_history"
)

# Create graph builder with our state
graph_builder = StateGraph(State)

# Create search tool and LLM
search_tool = TavilySearchResults(max_results=2)
llm = ChatGroq(
    temperature=0.1,
    model_name="mixtral-8x7b-32768",
)

# Bind the tool to the LLM
llm_with_tools = llm.bind_tools([search_tool])

# Modified chatbot node that uses memory
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
                # Update the last human message with AI response
                memory.save_context(
                    {"input": current_messages[-2].content},
                    {"output": last_message.content}
                )

    # Get recent history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # Create new context with only recent history
    limited_context = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            limited_context.append(("user", msg.content))
        elif isinstance(msg, AIMessage):
            limited_context.append(("assistant", msg.content))
    
    # Add the current message if it exists
    if current_messages and hasattr(current_messages[-1], "content"):
        limited_context.append(
            ("user", current_messages[-1].content)
        )

    # Get response using limited context
    response = llm_with_tools.invoke(limited_context)
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

# Function to run agent with memory status
def run_agent(user_input: str, debug: bool = True):
    events = graph.stream(
        {"messages": [("user", user_input)]},
        stream_mode="values"
    )
    
    if debug:
        print("\n=== Memory Status ===")
        print("Current memory contents:")
        chat_history = memory.load_memory_variables({})["chat_history"]
        for i, msg in enumerate(chat_history, 1):
            print(f"{i}. {msg.type}: {msg.content[:100]}...")
    
    for event in events:
        if "messages" in event:
            message = event["messages"][-1]
            if hasattr(message, "content"):
                print(f"\nAssistant: {message.content}")
            else:
                print(f"\nAssistant used tool: {message.tool_calls[0]['name']}")
                if debug:
                    print(f"Tool args: {message.tool_calls[0]['args']}")

# Test the agent
if __name__ == "__main__":
    print("Chat with the agent (type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        run_agent(user_input, debug=True)
