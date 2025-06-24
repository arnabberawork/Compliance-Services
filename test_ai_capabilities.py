import asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["api_gateway.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

async def test_ai_capabilities():
    # Connect to the compliance service using stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized")

            # Test AI capabilities
            print("\nChecking AI capabilities...")
            capabilities = await session.read_resource("capabilities://ai")
            print(capabilities)

if __name__ == "__main__":
    asyncio.run(test_ai_capabilities())
