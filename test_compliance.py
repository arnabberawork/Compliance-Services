import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection to the compliance service
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["compliance_service.py"],  # Connect directly to compliance_service.py
    env=None,  # Optional environment variables
)

async def test_compliance_service():
    """Test the compliance service directly without going through the API gateway"""
    print("Testing Compliance Service...")
    
    # Connect to the Compliance Service using stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized successfully")
            
            # Get server info
            server_info = await session.get_server_info()
            print(f"Connected to: {server_info.name} (version: {server_info.version})")
            
            # List available prompts
            prompts = await session.list_prompts()
            print("\nAvailable prompts:")
            for prompt in prompts:
                print(f"- {prompt}")
            
            # Get a prompt
            if "compliance_analysis_prompt" in [p.name for p in prompts]:
                print("\nTesting compliance analysis prompt...")
                prompt = await session.get_prompt(
                    "compliance_analysis_prompt",
                    arguments={
                        "expense_id": "EXP123",
                        "amount": 250.00,
                        "category": "travel"
                    }
                )
                print("Prompt result:")
                print(prompt)
            
            # List available resources
            resources = await session.list_resources()
            print("\nAvailable resources:")
            for resource in resources:
                print(f"- {resource}")
            
            # Check workflow status
            if "workflow://status" in [r.name for r in resources]:
                print("\nChecking workflow status...")
                status = await session.read_resource("workflow://status")
                print("Workflow status:")
                print(status)
            
            # Check service health
            if "health://service" in [r.name for r in resources]:
                print("\nChecking service health...")
                health = await session.read_resource("health://service")
                print("Service health:")
                print(health)
            
            # List available tools
            tools = await session.list_tools()
            print("\nAvailable tools:")
            for tool in tools:
                print(f"- {tool.name}: {tool.description}")
            
            # Test quick compliance check
            print("\nTesting quick_ai_assessment tool...")
            try:
                result = await session.call_tool(
                    "quick_ai_assessment",
                    {
                        "expense_id": "EXP123",
                        "amount": 250.00,
                        "category": "travel",
                        "vendor": "Hilton Hotels",
                        "employee_name": "John Doe",
                        "department": "Sales",
                        "description": "Hotel stay during client meeting",
                        "receipt_attached": True
                    }
                )
                print("Quick assessment result:")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error calling quick_ai_assessment: {e}")
            
            # Test full compliance analysis
            print("\nTesting analyze_compliance_with_ai tool...")
            try:
                full_result = await session.call_tool(
                    "analyze_compliance_with_ai",
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
                        "department": "Sales",
                        "full_analysis": True
                    }
                )
                print("Full analysis result:")
                print(json.dumps(full_result, indent=2))
            except Exception as e:
                print(f"Error calling analyze_compliance_with_ai: {e}")

if __name__ == "__main__":
    asyncio.run(test_compliance_service())

