from langchain_core.tools import tool
from typing import List
import operator
import asyncio
import threading
import numpy as np
import time
import random
import os
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
import pandas as pd
from crawl4ai import AsyncWebCrawler, BrowserConfig
import json
import requests
import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv

matplotlib.use('Agg')  # Force non-GUI backend
load_dotenv()

repl = PythonREPL()

@tool
def calculator(number_1: float, number_2: float, operation: str = "add") -> float:
    """Performs an arithmetic operation on two numbers.
    
    Args:
        number_1 (float): The first number (can be an integer or decimal).
        number_2 (float): The second number (can be an integer or decimal).
        operation (str, optional): The operation to perform ('add', 'subtract', 'multiply', 'divide'). Defaults to 'add'.
    Returns:
        float: The result of the operation
    Raises:
        ValueError: If an unsupported operation is provided
        ZeroDivisionError: If division by zero is attempted
    """
    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv
    }
    
    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}. Supported operations: {list(operations.keys())}")
    
    result = operations[operation](number_1, number_2)
    return result

@tool
def get_latest_quote(ticker: str) -> str:
    """Retrieve the latest stock quote price for a given symbol using Alpha Vantage API.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'IBM').
    Returns:
        str: The latest price as a string (e.g., '149.3100'), or an error message if the request fails.
    """

    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        
        # Extract the latest price from the response
        quote_data = data.get("Global Quote", {})
        latest_price = quote_data.get("05. price")
        
        if latest_price:
            return latest_price
        else:
            return "Error: Could not retrieve the latest price. Check the symbol or API response."
    
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch quote for {symbol}. {str(e)}"

@tool
def python_repl_tool(
    code: Annotated[str, "The Python code to execute to generate your chart using Matplotlib. Save with plt.savefig()."]
) -> str:
    """Execute Python code to generate Matplotlib charts. Save plots with `plt.savefig('filename.png')` and print 'Chart saved as filename.png'."""
    try:
        result = repl.run(code)
        plt.close('all')  # Clear figures to avoid memory issues
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nOutput: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."


    # If statement_type is not provided, use the default URL (income statement).
@tool
async def scrape_url(ticker: str, statement_type: str = None) -> str:
    """Run the scrape and return the results.
    Synchronous wrapper for the async scraper tool.
    
    Args:
        ticker (str): The ticker symbol for the company
        statement_type (str, optional): The page to retrieve (e.g., 'balance-sheet', 'snapshot', 'cash-flow-statement', 'income-statement', 'ratios')
    Returns:
        str: The scraped content in markdown format
    """
    
    # Construct URL based on statement_type
    if statement_type == "income statement" or statement_type == "income-statement" or statement_type is None:
        url = f"https://stockanalysis.com/stocks/{ticker}/financials"
    elif statement_type == "ratios":
        url = f"https://stockanalysis.com/stocks/{ticker}/financials/ratios"
    elif statement_type == "snapshot":
        url = f"https://stockanalysis.com/stocks/{ticker}/statistics/"
    else:
        url = f"https://stockanalysis.com/stocks/{ticker}/financials/{statement_type}/"
  
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

@tool
def number_summer(numbers: Annotated[List[float], "List of numbers to sum"]) -> float:
    """Sum a list of numbers."""
    return np.sum(numbers)
