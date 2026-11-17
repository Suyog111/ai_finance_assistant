# Claude Desktop MCP Integration Guide
### Connect AI Finance Assistant Tools to Claude Desktop via Model Context Protocol

This project includes a fully compliant **Model Context Protocol (MCP)** server that exposes financial research tools directly into Anthropic's Claude Desktop app.

---

## 🛠️ Exposed Tools

When connected, Claude Desktop gains access to five native financial tools:

| Tool Name | Parameters | Purpose |
|---|---|---|
| `get_stock_quote` | `ticker: str` | Real-time price, day change, 52W range, P/E ratio, market cap (US & Indian stocks like `AAPL`, `RELIANCE.NS`, `TCS.NS`). |
| `get_market_indices` | `market: str = "all"` | Real-time benchmark index levels for S&P 500, NASDAQ, Dow, Nifty 50, and Sensex. |
| `analyze_portfolio` | `holdings_json: str` | Calculates total value, cost basis, unrealized P&L %, sector distribution, and 0–100 diversification score. |
| `calculate_goal_savings` | `target_amount, years, annual_return_pct, currency` | Calculates required monthly savings / SIP and recommended stock/bond/cash asset allocation. |
| `explain_financial_concept` | `concept: str, category: str` | Queries the local FAISS vector knowledge base (67 articles) for verified educational concepts with citations. |

---

## ⚙️ Configuration Setup for Claude Desktop

1. Open Claude Desktop's configuration file:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the `ai_finance_assistant` MCP server configuration:

```json
{
  "mcpServers": {
    "ai_finance_assistant": {
      "command": "python3",
      "args": [
        "/Users/suyogniwane/Documents/workspace/IKWorkspace/IK Capstone Projects/ai_finance_assistant/src/mcp_server/server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/suyogniwane/Documents/workspace/IKWorkspace/IK Capstone Projects/ai_finance_assistant"
      }
    }
  }
}
```

3. Restart Claude Desktop. You will see a hammer icon (`🛠️`) in the prompt input confirming that the 5 financial tools are active!

---

## 💬 Sample Prompts to Try in Claude Desktop
- *"What is Reliance Industries (RELIANCE.NS) and Apple (AAPL) trading at today?"*
- *"Analyze my portfolio: I hold 10 shares of AAPL at 150 and 20 shares of RELIANCE.NS at 2850."*
- *"How much should I save monthly to reach ₹50 Lakhs in 10 years at 12% return?"*
- *"Explain the new Indian capital gains tax rules after Budget 2024."*
