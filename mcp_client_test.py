"""Smoke test for mcp_server.py: start it over stdio, list tools, run one search."""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    """Launch the MCP server as a subprocess and call search_acba_reports once."""
    # sys.executable ensures the server runs in the same virtual environment.
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "search_acba_reports", {"question": "net profit 2025"}
            )
            print(result.content[0].text[:500])


if __name__ == "__main__":
    asyncio.run(main())
