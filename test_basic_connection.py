"""
Basic connection test to verify MCP service is working without AI features.
"""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Temporarily disable AI to test basic connectivity
os.environ["ENABLE_AI"] = "false"
os.environ["OPENAI_API_KEY"] = ""

async def test_basic_connection():
    """Test basic MCP connection without AI features"""
    print("Testing basic MCP connection (AI disabled)...")
    
    server_params = StdioServerParameters(
        command="python",
        args=["api_gateway.py"],
        env=None,
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Session initialized successfully")
                
                # Test basic functionality
                print("\n📋 Testing list prompts...")
                prompts = await session.list_prompts()
                print(f"Found {len(prompts.prompts)} prompts")
                
                print("\n🔧 Testing list tools...")
                tools = await session.list_tools()
                print(f"Found {len(tools.tools)} tools")
                
                print("\n📊 Testing list resources...")
                resources = await session.list_resources()
                print(f"Found {len(resources.resources)} resources")
                
                print("\n🏥 Testing health resource...")
                health = await session.read_resource("health://gateway")
                print("Health check completed")
                
                print("\n✅ All basic tests passed!")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_basic_connection())