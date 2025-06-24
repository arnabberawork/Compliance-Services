# MCP Compliance Services

A modern, AI-powered compliance analysis system built on the Model Context Protocol (MCP).

## Overview

This project transforms traditional REST API-based compliance services into MCP servers, providing a standardized way for AI models to interact with compliance analysis tools. The system uses LangGraph for workflow orchestration and OpenAI's models for intelligent compliance analysis.

## Features

- **AI-Powered Compliance Analysis**: Intelligent assessment of expenses for policy compliance
- **MCP Protocol Support**: Standardized interface for AI assistants to discover and use tools
- **LangGraph Workflows**: Sophisticated orchestration of compliance analysis steps
- **Multiple Analysis Modes**: Quick assessments, comprehensive analysis, and batch processing
- **Explainable AI**: Transparency into AI decision-making processes

## Quick Start

### Prerequisites

- Python 3.12 or higher
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mcp-compliance.git
   cd mcp-compliance
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

### Running the Services

1. Start the Compliance Service:
   ```bash
   python compliance_service.py
   ```

2. Start the API Gateway:
   ```bash
   python api_gateway.py
   ```

Alternatively, use the provided script to start both services:
```bash
python start_services.py
```

### Testing

Run the test scripts to verify functionality:

```bash
# Test the compliance service directly
python test_gateway.py
python test_compliance.py

# Test tools, resources, and prompts through the API gateway
python test_tools.py
python test_resources.py
python test_prompts.py
```

## Using with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "compliance-gateway": {
      "command": "python",
      "args": ["/path/to/api_gateway.py"]
    },
    "ai-compliance": {
      "command": "python",
      "args": ["/path/to/compliance_service.py"]
    }
  }
}
```

## Using Programmatically

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to the service
async with stdio_client({"command": "python", "args": ["./api_gateway.py"]}) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Call a tool
        result = await session.call_tool(
            "analyze_compliance",
            {
                "expense_id": "EXP123",
                "employee_id": "EMP456",
                "amount": 250.00,
                "category": "travel",
                # ... other parameters
            }
        )
```

## Documentation

- [Architecture Overview](architecture.md) - System design and components
- [Implementation Guide](mcp-implementation-guide-summary.md) - Details on MCP implementation
- [Setup Guide](setup-guide.md) - Detailed setup instructions

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_API_KEY | Your OpenAI API key | (Required) |
| LLM_MODEL | AI model to use | gpt-4o-mini |
| AI_TEMPERATURE | Temperature for AI responses | 0.1 |
| AI_MAX_TOKENS | Maximum tokens for AI responses | 4000 |
| ENABLE_AI | Enable/disable AI features | true |
| LOG_LEVEL | Logging level | INFO |

## License

[MIT License](LICENSE)

