from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
import asyncio
import re
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from alpaca_trade_api.rest import REST
from datetime import datetime
from io import StringIO
import sys

from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.agents.react import ReActAgent

from fastapi.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv()


def all_finished():
    return all(status == "Finished" for status in agent_status.values())


agent_status = {
    "portfolio": "Idle",
    "research": "Idle",
    "sentiment": "Idle",
    "decision": "Idle"
}

def update_status(agent: str, status: str):
    agent_status[agent] = status
    # log(f"[STATUS] {agent.capitalize()} Agent: {status}")



def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

DRY_RUN = False

alpaca = REST(
    key_id=os.getenv("AlpacaApiKey"),
    secret_key=os.getenv("AlpacaSecret"),
    base_url="https://paper-api.alpaca.markets"
)

e2b_tool = E2BInterpreterTool(
    connection=E2BConnection(api_key=os.getenv("E2BKey"))
)

llm = OpenAI(
    id="openai",
    connection=OpenAIConnection(api_key=os.getenv("ChatGptKey")),
    model="gpt-4o",
    temperature=0.3,
    max_tokens=2000,
)

portfolio_agent = ReActAgent(
    name="PortfolioAgent",
    llm=llm,
    tools=[e2b_tool],
    role="You are a Portfolio Evaluator. Review current positions, compare with historical performance, and decide if each asset should be held or sold.",
    max_loops=10,
)

stock_researcher = ReActAgent(
    name="StockResearcher",
    llm=llm,
    tools=[e2b_tool],
    role="You're a Stock Research Analyst. Use Python code and live data to assess potential of new stocks and analyze macroeconomic trends. Return tickers with reasoning.",
    max_loops=20,
)

sentiment_agent = ReActAgent(
    name="SentimentAgent",
    llm=llm,
    tools=[e2b_tool],
    role="You're a Sentiment & News Analyst. Scrape news, Reddit, or Twitter for public opinion on given stocks and major events. Summarize for buy/sell signals.",
    max_loops=20,
)

decision_agent = ReActAgent(
    name="PrioritizerAgent",
    llm=llm,
    tools=[e2b_tool],
    role="You're a Portfolio Manager. Combine: Current cash, Portfolio performance, Research picks, Sentiment signals. Then decide which to HOLD, SELL, or BUY and for how much. Format like: 'Buy $500 of AAPL' or 'Sell $400 of TSLA'.",
    max_loops=15,
)

async def run_stock_broker_team():
    account = alpaca.get_account()
    cash_available = float(account.cash)
    holdings = alpaca.list_positions()
    open_orders = alpaca.list_orders(status='open')

    holdings_summary = "\n".join([f"{pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price}" for pos in holdings])
    orders_summary = "\n".join([f"{order.symbol}: {order.side} {order.qty} shares" for order in open_orders])

    update_status("portfolio", "Initializing...")
    await asyncio.sleep(0.1)
    update_status("portfolio", "Running...")
    await asyncio.sleep(0.1)
    portfolio_review = await portfolio_agent.run({
        "input": f"Today's portfolio:\n{holdings_summary}\n\nOpen Orders:\n{orders_summary}\n\nAssess performance from yesterday and decide which to sell, hold, or trim."
    })
    update_status("portfolio", "Finished")

    # --- Research Agent ---
    update_status("research", "Initializing...")
    await asyncio.sleep(0.1)
    update_status("research", "Running...")
    await asyncio.sleep(0.1)
    research_result = await stock_researcher.run({
        "input": f"We have ${cash_available} in free cash. Are there any strong new stocks or macroeconomic trends like tariffs that suggest we should or should not buy today?"
    })
    update_status("research", "Finished")

    # --- Sentiment Agent ---
    update_status("sentiment", "Initializing...")
    await asyncio.sleep(0.1)
    update_status("sentiment", "Running...")
    await asyncio.sleep(0.1)
    sentiment_res = await sentiment_agent.run({
        "input": f"Research said: {research_result.output['content']}. Assess market sentiment and macro risks like tariffs. Should we buy these or wait?"
    })
    update_status("sentiment", "Finished")

    # --- Decision Agent ---
    update_status("decision", "Initializing...")
    await asyncio.sleep(0.1)
    update_status("decision", "Running...")
    await asyncio.sleep(0.1)
    final_res = await decision_agent.run({
        "input": f"Cash: ${cash_available}\nPortfolio:\n{holdings_summary}\n\nOrders:\n{orders_summary}\n\nPortfolio Review:\n{portfolio_review.output['content']}\n\nResearch:\n{research_result.output['content']}\n\nSentiment:\n{sentiment_res.output['content']}\n\nWhat to do today?"
    })
    update_status("decision", "Finished")

    final_res = await decision_agent.run({
        "input": f"Cash: ${cash_available}\nPortfolio:\n{holdings_summary}\n\nOrders:\n{orders_summary}\n\nPortfolio Review:\n{portfolio_review.output['content']}\n\nResearch:\n{research_result.output['content']}\n\nSentiment:\n{sentiment_res.output['content']}\n\nWhat to do today?"
    })

    update_status("decision", "Finished")

    # Final summary
    decision_output = final_res.output["content"]
    log("\nFinal Investment Plan:\n" + decision_output)

    with open("trade_summary.txt", "w") as f:
        f.write("Final Investment Plan:\n")
        f.write(decision_output + "\n")

    # Execute trades if necessary
    for line in decision_output.splitlines():
        if line.lower().startswith("buy") or line.lower().startswith("sell"):
            try:
                matches = re.findall(r"(Buy|Sell) \$([\d,]+).*?([A-Z]{1,5})", line, re.IGNORECASE)
                for action, amount_str, symbol in matches:
                    amount = float(amount_str.replace(",", ""))
                    try:
                        price = alpaca.get_latest_trade(symbol).price
                    except Exception as e:
                        log(f"Could not get latest price for {symbol}: {e}")
                        continue
                    qty = round(amount / price, 2)
                    log(f"Preparing to {action.upper()} {qty} shares of {symbol} (~${amount})")
                    if DRY_RUN:
                        log(f"[DRY RUN] Would {action.upper()} {qty} shares of {symbol} (~${amount})")
                    else:
                        side = action.lower()
                        order = alpaca.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            type="market",
                            time_in_force="gtc"
                        )
                        log(f"{action.upper()} order placed for {symbol}: {qty} shares (~${amount}) (Order ID: {order.id})")
            except Exception as e:
                log(f"Failed to parse or execute order: '{line}' - {e}")
    return decision_output


app = FastAPI()


app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    return JSONResponse(content=agent_status)


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return FileResponse("index.html")

@app.get("/run")
async def run_system():
    for agent in agent_status:
        agent_status[agent] = "Idle"

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        await run_stock_broker_team()
    finally:
        sys.stdout = old_stdout
    return mystdout.getvalue()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
