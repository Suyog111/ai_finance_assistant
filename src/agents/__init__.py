"""
src/agents/__init__.py
Exports all six specialized agents and the base agent interface.
"""

from src.agents.base_agent import BaseFinanceAgent
from src.agents.goal_agent import GoalPlanningAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.news_agent import NewsSynthesizerAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.qa_agent import FinanceQAAgent
from src.agents.tax_agent import TaxEducationAgent

__all__ = [
    "BaseFinanceAgent",
    "FinanceQAAgent",
    "PortfolioAnalysisAgent",
    "MarketAnalysisAgent",
    "GoalPlanningAgent",
    "NewsSynthesizerAgent",
    "TaxEducationAgent",
]
