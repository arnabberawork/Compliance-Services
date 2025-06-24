# Quick Setup Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure Environment

1. Copy the `.env` file to your project directory
2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

## 3. Verify Installation

```bash
# Test that MCP is installed
python -c "import mcp; print('MCP installed successfully')"

# Test that AI libraries are installed
python -c "import langchain_openai; print('LangChain installed successfully')"
```

## 4. Run the Services

```bash
# Terminal 1 - Start Compliance Service
python compliance_service.py

# Terminal 2 - Start Gateway
python api_gateway.py
```

## Minimal Requirements Explained

- **mcp**: Core MCP protocol implementation
- **langchain-openai**: OpenAI integration for AI features
- **langchain-core**: Core LangChain components (prompts, messages)
- **langgraph**: Workflow orchestration for compliance analysis
- **python-dotenv**: Load environment variables from .env file
- **httpx**: Async HTTP client for service communication
- **typing-extensions**: TypedDict support for Python < 3.12

## Notes

- The `.env` file only includes the essential `OPENAI_API_KEY` and basic AI settings
- Other environment variables have sensible defaults and are optional
- If you don't have an OpenAI API key, the services will run but AI features will be disabled
- For production, consider adding more specific version pins in requirements.txt