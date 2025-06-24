import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["api_gateway.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

async def test_prompts():
    # Connect to the service using stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized")
            
            # List available prompts
            print("\nListing available prompts...")
            prompts_response = await session.list_prompts()

            for prompt in prompts_response.prompts:
                print(f"Name - {prompt.name}")
                print(f"Description - {prompt.description}")
                print("====="*30)
            
            # Test expense analysis prompt
            print("\nTesting expense analysis prompt...")
            prompt = await session.get_prompt(
                "expense_analysis_prompt",
                arguments={
                    "amount": "250.00",
                    "category": "travel",
                    "vendor": "Hilton Hotels",
                    "department": "Sales"
                }
            )
            print("Expense analysis prompt:")
            print(prompt)
            
            # Test quick check prompt
            print("\nTesting quick check prompt...")
            quick_prompt = await session.get_prompt(
                "quick_check_prompt",
                arguments={"expense_id": "EXP123"}
            )
            print("Quick check prompt:")
            print(quick_prompt)

if __name__ == "__main__":
    asyncio.run(test_prompts())