"""
src/mcp_server/__init__.py
Exports the Model Context Protocol (MCP) server.
"""

from src.mcp_server.server import main, server

__all__ = ["server", "main"]
