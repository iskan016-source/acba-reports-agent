"""MCP server exposing the Acba reports search as a tool for MCP clients.

Run with:  python mcp_server.py   (stdio transport)
"""

from mcp.server.mcpserver import MCPServer

from bank_agent.search import search

server = MCPServer("acba-reports")


@server.tool()
def search_acba_reports(question: str) -> list[dict]:
    """Search Acba Bank's 2025 Annual Report and audited IFRS financial statements.
    Returns the 5 most relevant passages, each with source, page, score and text.
    Financial-statement figures are usually in thousands of AMD."""
    return search(question, top_k=5)


if __name__ == "__main__":
    # stdio transport: the client launches this process and talks over stdin/stdout.
    server.run()
