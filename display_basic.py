
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# First, define what state we want to store
# First, define what state we want to store
# First, define what state we want to store
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create graph builder with our state
graph_builder = StateGraph(State)

# Create search tool and LLM
search_tool = TavilySearchResults(max_results=2)
llm = ChatGroq(
    temperature=0.1,
    model_name="mixtral-8x7b-32768",  # You can also use "llama2-70b-4096"
)

# Bind the tool to the LLM so it knows how to use it
llm_with_tools = llm.bind_tools([search_tool])
print(llm_with_tools)

# Define the chatbot node that will generate responses
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add nodes to graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[search_tool])
graph_builder.add_node("tools", tool_node)

# Add edges - this defines how nodes are connected
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Add memory checkpointer
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer
memory = MemorySaver()

# Compile the graph with checkpointer
graph = graph_builder.compile(checkpointer=memory)

# Visualize the graph structure
def visualize_graph():
    try:
        from graphviz import Digraph

        # Create a new directed graph
        dot = Digraph(comment='Agent Graph')
        dot.attr(rankdir='LR')  # Left to right direction

        # Add nodes
        nodes = ["START", "chatbot", "tools", "END"]
        for node in nodes:
            dot.node(node, node)

        # Add edges
        dot.edge("START", "chatbot")
        dot.edge("chatbot", "tools", "needs tool")
        dot.edge("tools", "chatbot", "tool result")
        dot.edge("chatbot", "END", "direct response")

        # Print in text format
        print("\nGraph Structure:")
        print("Nodes:", ["START", "chatbot", "tools", "END"])
        print("Edges:")
        print("- START -> chatbot")
        print("- chatbot -> tools (when tool needed)")
        print("- tools -> chatbot (with tool result)")
        print("- chatbot -> END (direct response)")

        try:
            # Try to render if in a Jupyter environment
            from IPython.display import display
            display(dot)
        except:
            # Fallback to simple text representation
            pass

    except ImportError:
        print("\nGraph Structure (Text):")
        print("START -> chatbot -> [tools -> chatbot | END]")
        print("\nFlow:")
        print("1. Start at chatbot")
        print("2. If tool needed: chatbot -> tools -> chatbot")
        print("3. If no tool needed: chatbot -> END")
        print("\nComponents:")
        print("- chatbot: Processes user input and generates responses")
        print("- tools: Executes search and other external actions")

# Show the visualization
visualize_graph()

# Function to run the agent
def run_agent(user_input: str, thread_id: str = "1", debug: bool = True):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get initial state
    if debug:
        print("\n=== Starting Agent Execution ===")
        initial_state = graph.get_state(config)
        print(f"Initial State - Messages: {len(initial_state.values.get('messages', []))}")
    
    events = graph.stream(
        {"messages": [("user", user_input)]}, 
        config, 
        stream_mode="values"
    )
    
    step = 0
    
    # Process responses
    for event in events:
        if debug:
            step += 1
            print(f"\nStep {step}:")
            if "next" in event:
                print(f"Next node: {event['next']}")
            
        if "messages" in event:
            message = event["messages"][-1]
            if hasattr(message, "content"):
                print(f"Assistant: {message.content}")
            else:
                print(f"Assistant used tool: {message.tool_calls[0]['name']}")
                if debug:
                    print(f"Tool args: {message.tool_calls[0]['args']}")
    
    if debug:
        # Get final state
        final_state = graph.get_state(config)
        print("\n=== Execution Complete ===")
        print(f"Final State - Messages: {len(final_state.values.get('messages', []))}")
        print("=" * 50)

# Example usage
if __name__ == "__main__":
    print("Chat with the agent (type 'exit' to quit)")
    thread_id = "1"
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        run_agent(user_input, thread_id)
