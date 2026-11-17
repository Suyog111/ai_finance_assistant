# Capstone Demo Video Walkthrough Script (5–10 Minutes)
### AI Finance Assistant: Democratizing Financial Literacy Through Intelligent Multi-Agent AI

---

## 🎬 Video Recording Blueprint Overview

| Section | Target Duration | Tab / Interface | Goal |
|---|:---:|---|---|
| **1. Intro & Architecture** | 1:00 | Slides / Streamlit Sidebar | Introduce project vision, multi-agent architecture, and dual-market support (US & India). |
| **2. Multi-Turn Chat & RAG** | 2:00 | `💬 Chat & Advisory` | Demonstrate intelligent intent routing, RAG retrieval citations, and context preservation. |
| **3. Tax Education (US & India)** | 1:30 | `💬 Chat & Advisory` | Showcase Budget 2024 Indian capital gains and US 401(k)/IRA rules with mandatory disclaimers. |
| **4. Portfolio Analysis** | 2:00 | `📊 Portfolio Dashboard` | Walk through live holdings (US + India), Plotly charts, diversification score, and AI review. |
| **5. Real-Time Market Overview** | 1:30 | `📈 Live Market Overview` | Show Nifty 50, Sensex, S&P 500, live stock search (AAPL, RELIANCE.NS), and recent news feed. |
| **6. Goal Planner & Compounding** | 1:30 | `🎯 Goal Planner` | Demonstrate SIP / monthly savings math, interactive stacked area growth chart, and AI roadmap. |
| **7. Code Quality & Conclusion** | 0:30 | Terminal (`pytest`) | Run `pytest` showing 113 passing tests and 87% test coverage. |
| **Total** | **~10:00** | | |

---

## Detailed Step-by-Step Script

### Section 1: Introduction (0:00 – 1:00)
- **Visual:** Open Streamlit app at `http://localhost:8501`. Show the sidebar with system status and user preferences.
- **Narrative:**
  > *"Hello everyone! Welcome to the demonstration of my capstone project: the **AI Finance Assistant**, built to democratize financial literacy through production-ready multi-agent AI.*
  >
  > *Financial jargon, complex tax rules, and market volatility often intimidate everyday individuals. To solve this, our system coordinates six specialized agents using **LangGraph**, grounds responses using **FAISS vector retrieval** over a 67-article financial knowledge base, and connects to live market data for both **US and Indian markets**."*

---

### Section 2: Conversational Multi-Turn Chat (1:00 – 3:00)
- **Visual:** In Tab 1 (`💬 Chat & Advisory`).
- **Action 1:** Click on the starter prompt:
  - *"How does the Three-Fund Portfolio work for beginners?"*
- **Explain:**
  > *"Notice how the system instantaneously routes this query to the **Finance Q&A Agent**. It synthesizes a clear explanation of VTI, VXUS, and BND, cites the underlying knowledge base sources in an expandable drawer, and includes our mandatory educational disclaimer."*
- **Action 2:** Ask a multi-turn follow-up question:
  - *"How does this strategy compare to dollar-cost averaging?"*
- **Explain:**
  > *"Because our LangGraph graph maintains conversation state memory using `MemorySaver`, the agent understands the context of our previous discussion without needing to re-explain the portfolio structure."*

---

### Section 3: Tax Education — US & Indian Frameworks (3:00 – 4:30)
- **Visual:** Still in Tab 1 (`💬 Chat & Advisory`).
- **Action 1:** Type:
  - *"What are the new capital gains tax rules in India after Budget 2024 for stocks and mutual funds?"*
- **Explain:**
  > *"The intent router immediately recognizes this as a tax query and delegates it to the **Tax Education Agent**. It accurately highlights Section 111A (STCG at 20%) and Section 112A (LTCG at 12.5% with the new ₹1.25 Lakh exemption threshold), and appends our specialized tax disclaimer."*
- **Action 2:** Type:
  - *"What about US tax-advantaged accounts like Roth IRAs and 401(k)s?"*
- **Explain:**
  > *"The agent effortlessly handles cross-border financial concepts, explaining pre-tax vs. post-tax growth and employer matching."*

---

### Section 4: Portfolio Analysis Dashboard (4:30 – 6:30)
- **Visual:** Click on Tab 2 (`📊 Portfolio Dashboard`).
- **Demonstration:**
  1. Highlight the 4 KPI cards: **Total Current Value**, **Cost Basis**, **Unrealized P&L**, and the **0–100 Diversification Score**.
  2. Point out that the holdings table tracks real-time prices for both US equities (`AAPL`, `MSFT`, `VTI`) and Indian equities (`RELIANCE.NS`, `TCS.NS`).
  3. Explore the **Plotly Sector Donut Chart** and the **Unrealized Return Bar Chart**.
  4. Demonstrate adding a new holding: enter `NVDA`, 5 shares at $120.00. Show how the table and charts immediately update.
  5. Click **"Generate Comprehensive AI Portfolio Review"**:
     > *"The Portfolio Analysis Agent evaluates concentration risks, flags over-allocated sectors, and suggests rebalancing strategies."*

---

### Section 5: Real-Time Market Overview (6:30 – 8:00)
- **Visual:** Click on Tab 3 (`📈 Live Market Overview`).
- **Demonstration:**
  1. Show the **Major Indices** grid with real-time quotes: Nifty 50, BSE Sensex, Bank Nifty, S&P 500, NASDAQ, Dow Jones.
  2. Toggle between *All Markets*, *Indian Markets*, and *US Markets*.
  3. In the ticker lookup, search for `RELIANCE.NS` (or `AAPL`).
  4. Showcase the live price card: Day Change %, P/E Ratio, Market Cap, and 52-Week Range.
  5. Scroll down to show the **Live Financial Headlines Feed** pulled directly from the wire.

---

### Section 6: Goal Planning & Compounding Simulator (8:00 – 9:30)
- **Visual:** Click on Tab 4 (`🎯 Goal Planner`).
- **Demonstration:**
  1. Set Goal Name to *"House Down Payment"*, Currency to *INR (₹)*, and Target Amount to *₹50,00,000 (50 Lakhs)* over a *10-year horizon*.
  2. Adjust the expected return slider to 12% and inflation to 6%.
  3. Highlight the KPI card: the exact **Required Monthly SIP** needed.
  4. Hover over the **Plotly Stacked Area Chart** demonstrating how compound interest outpaces principal invested over time.
  5. Show the **Recommended Asset Allocation Bar** adjusting between stocks, bonds, and cash based on the horizon slider.
  6. Click **"Generate AI Milestone Roadmap"** to show the structured action plan.

---

### Section 7: Code Quality, Testing & Wrap-up (9:30 – 10:00)
- **Visual:** Switch to terminal.
- **Action:** Run:
  ```bash
  python3 -m pytest tests/ -v
  ```
- **Narrative:**
  > *"To ensure enterprise-grade reliability, our codebase includes a comprehensive test suite of 113 unit and integration tests with **87% overall coverage**, completely passing all `ruff` and `black` quality checks.*
  >
  > *Thank you for watching the demo of the AI Finance Assistant!"*
