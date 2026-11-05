import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # How to start the server: run "python mcp_server.py"
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    async with stdio_client(params) as (read, write):          # start server, open the pipes
        async with ClientSession(read, write) as session:      # MCP conversation
            await session.initialize()                         # handshake

            tools = await session.list_tools()                 # "what can you do?"
            print("Tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "search_acba_reports", {"question": "net profit 2025"}
            )
            print(result.content[0].text[:500])                # first 500 characters


asyncio.run(main())