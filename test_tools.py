import asyncio,json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["api_gateway.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

async def test_tools():
    # Connect to the service using stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized")

            # List available tools
            print("\nListing available tools...")
            tools_response = await session.list_tools()

            for tool in tools_response.tools:
                print(f"Name - {tool.name}")
                print(f"Description - {tool.description}")
                print("====="*30)
            
            # Test quick compliance check tool
            print("\nTesting quick compliance check tool...")
            result = await session.call_tool(
                "quick_compliance_check",
                {
                    "expense_id": "EXP123",
                    "amount": "250.00",
                    "category": "travel",
                    "vendor": "Hilton Hotels",
                    "receipt_attached": True
                }
            )
            result = result.content
            # Step 1: Get the text from the first item
            json_text = result[0].text

            # Step 2: Convert the JSON string to a dictionary
            parsed_result = json.loads(json_text)

            # Step 3: Print or use it
            for key, value in parsed_result.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_tools())