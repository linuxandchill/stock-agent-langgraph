import os
import numpy as np
import time
import random
import asyncio
import threading
import whisper
import sounddevice as sd
from typing import List
import operator
from dotenv import load_dotenv
from queue import Queue
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from rich.console import Console
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_aws import ChatBedrockConverse
from langchain_community.tools.tavily_search import TavilySearchResults
import custom_tools

load_dotenv()
console = Console()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
memory = MemorySaver()


search_tool = TavilySearchResults(max_results=3,
    include_answer=True,
    include_raw_content=True,
    search_depth="advanced",
    )

tools = [
    custom_tools.scrape_url, 
    custom_tools.number_summer, 
    custom_tools.calculator, 
    custom_tools.get_latest_quote
    # search_tool, 
    ]

llm = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    temperature=0,
    max_tokens=5120, # nova
)
'''
llm = ChatBedrockConverse(
    # model="amazon.nova-pro-v1:0",
    # model="amazon.nova-micro-v1:0",
    # max_tokens=4096, 
    # model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0,
    max_tokens=4096, 
)
llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
'''

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    system_prompt = SystemMessage(content="""
                                  You are Warren, a helpful assistant. You are very capable and must be succinct and accurate in your answers. If a user's request lacks details needed to complete your answer you will ask for clarification or more information. If you can make a reasonable decision about what the user is requesting then make an attempt to complete the request. If a tool is available that could complete the user's request, you must use it.""")
    messages = [system_prompt] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "tyler1"}}

#  main loop to run asynchronously
async def main():
    while True:
        try:
            user_input = input(f">> ")
            events = graph.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="values",
            )
            async for event in events:
                last_message = event["messages"][-1]
                if last_message.type == "human":
                    pass
                elif last_message.type == "ai":
                    console.print(f"\n[bold cyan1]WARREN >> [bold cyan1]\n")
                    console.print(f"[light_goldenrod1]{last_message.content}[light_goldenrod1]")
        except KeyboardInterrupt:
            print("Shutting down...")
            break
        except Exception as e:
            print(f"BROKEN: {e}")
            break
if __name__ == "__main__":
    asyncio.run(main())
