import asyncio, json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["api_gateway.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

async def test_resources():
    # Connect to the service using stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized")
            
            # List available resources
            print("\nListing available resources...")
            resources_response = await session.list_resources()
            for resource in resources_response.resources:
                print(f"Name - {resource.name}")
                print(f"Description - {resource.description}")
                print("====="*30)
            
            # Test gateway health resource
            print("\nChecking gateway health...")
            health = await session.read_resource("health://gateway")
            result = health.contents
            # Step 1: Get the text from the first item
            json_text = result[0].text
            # Step 2: Convert the JSON string to a dictionary
            parsed_result = json.loads(json_text)
            # Step 3: Print or use it
            for key, value in parsed_result.items():
                print(f"{key}: {value}")
            
            # Test AI capabilities resource
            print("\nChecking AI capabilities...")
            capabilities = await session.read_resource("capabilities://ai")
            result = capabilities.contents
            # Step 1: Get the text from the first item
            json_text = result[0].text
            # Step 2: Convert the JSON string to a dictionary
            parsed_result = json.loads(json_text)
            # Step 3: Print or use it
            for key, value in parsed_result.items():
                print(f"{key}: {value}")
                

if __name__ == "__main__":
    asyncio.run(test_resources())