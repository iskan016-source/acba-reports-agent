from mcp.server.mcpserver import MCPServer
from bank_agent.search import search          # reuse your RAG search

server = MCPServer("acba-reports")            # the server's name, shown to clients


@server.tool()                                # this decorator makes the function an MCP tool
def search_acba_reports(question: str) -> list[dict]:
    """Search Acba Bank's 2025 Annual Report and audited IFRS financial statements.
    Returns the 5 most relevant passages, each with source, page, score and text.
    Financial-statement figures are usually in thousands of AMD."""
    # TODO: return the results of search(...) for this question, top 5
    return search(question, top_k = 5)


if __name__ == "__main__":
    server.run()      # default transport "stdio": talks to the client through standard input/output