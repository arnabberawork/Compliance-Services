import asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["api_gateway.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

async def test_api_gateway_service():
    # Connect to the API Gateway service using stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available prompts
            prompts = await session.list_prompts()
            print("Prompts :",prompts ) 
            print("====="*30 )

            # Get a prompt
            prompt = await session.get_prompt(
                "expense_analysis_prompt",
                arguments={
                    "amount": "250.00",
                    "category": "travel",
                    "vendor": "Hilton Hotels",
                    "department": "Sales"
                }
            )
            print("Prompt :",prompt )
            print("====="*30 )

            # Test another prompt
            quick_prompt = await session.get_prompt(
                "quick_check_prompt",
                arguments={"expense_id": "EXP123"}
            )
            print("Quick Check Prompt:", quick_prompt)
            print("====="*30 )

            # List available resources
            resources = await session.list_resources()
            print("Resources :",resources ) 
            print("====="*30 )

            # List available tools
            tools = await session.list_tools()
            print("Tools :",tools )
            print("====="*30 )
            
            # Test quick compliance check
            print("Testing quick compliance check...")
            # Call a tool
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
            
            print("Quick compliance check result:")
            print(result)

            # Test full compliance analysis
            print("\nTesting full compliance analysis...")
            full_result = await session.call_tool(
                "analyze_compliance",
                {
                    "expense_id": "EXP456",
                    "employee_id": "EMP001",
                    "employee_name": "John Doe",
                    "amount": 500.00,
                    "category": "travel",
                    "vendor": "Hilton Hotels",
                    "description": "Hotel stay during client meeting",
                    "date": "2024-06-24",
                    "receipt_attached": True,
                    "approval_status": "pending",
                    "department": "Sales"
                }
            )
            print("Full analysis result:")
            print(full_result)

            # Test gateway health status
            print("\nChecking gateway health...")
            health = await session.read_resource("health://gateway")
            print(health)

            # Test AI capabilities
            print("\nChecking AI capabilities...")
            capabilities = await session.read_resource("capabilities://ai")
            print(capabilities)

if __name__ == "__main__":
    asyncio.run(test_api_gateway_service())


