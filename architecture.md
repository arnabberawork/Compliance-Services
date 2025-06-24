# Architecture Overview

## System Architecture

The MCP Compliance Services system consists of several interconnected components that work together to provide AI-powered compliance analysis through the Model Context Protocol (MCP).

### High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   MCP Client    │◄───►│   API Gateway   │◄───►│  Compliance     │
│   (Claude/VSC)  │     │   (MCP Server)  │     │  Service        │
│                 │     │                 │     │  (MCP Server)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │  LangGraph      │
                                               │  Workflow       │
                                               │                 │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │  OpenAI API     │
                                               │  (LLM)          │
                                               │                 │
                                               └─────────────────┘
```

### Components

#### 1. MCP Clients

- **Claude Desktop**: AI assistant that can discover and use MCP tools
- **VS Code**: IDE with MCP integration for developer workflows
- **Custom Clients**: Any application that implements the MCP client protocol

#### 2. API Gateway (MCP Server)

- **Role**: Central entry point for all compliance analysis requests
- **Responsibilities**:
  - Tool discovery and registration
  - Request routing to appropriate services
  - AI orchestration for optimal processing strategy
  - Service health monitoring
  - Resource aggregation

#### 3. Compliance Service (MCP Server)

- **Role**: Core compliance analysis engine
- **Responsibilities**:
  - AI-powered compliance analysis
  - Quick assessments
  - Comprehensive analysis
  - Batch processing
  - Employee insights
  - Decision explanation

#### 4. LangGraph Workflow

- **Role**: Orchestrates the steps in compliance analysis
- **Components**:
  - Context gathering
  - Policy analysis
  - Pattern detection
  - Risk assessment
  - Recommendation generation
  - Escalation decision
  - Result finalization

#### 5. OpenAI API

- **Role**: Provides AI capabilities for analysis
- **Usage**:
  - Policy reasoning
  - Pattern detection
  - Risk assessment
  - Recommendation generation
  - Decision explanation

## Data Flow

### 1. Compliance Analysis Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │         │     │         │     │         │
│ Request │────►│ Context │────►│ Policy  │────►│ Pattern │────►│  Risk   │
│         │     │ Gather  │     │ Analysis│     │ Detect  │     │ Assess  │
│         │     │         │     │         │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
                                                                     │
                                                                     ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │         │     │         │     │         │
│ Result  │◄────│ Finalize│◄────│Escalate │◄────│ Generate│◄────│   AI    │
│         │     │ Result  │     │ Decision│     │ Recom.  │     │ Insights│
│         │     │         │     │         │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
```

### 2. MCP Protocol Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │         │     │         │
│ Client  │────►│ Tool    │────►│ Gateway │────►│ Service │
│ Request │     │ Call    │     │ Routes  │     │ Executes│
│         │     │         │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                                     │
                                                     ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │         │     │         │
│ Client  │◄────│ Response│◄────│ Gateway │◄────│ Service │
│ Receives│     │ Return  │     │ Formats │     │ Result  │
│         │     │         │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

## Key Design Decisions

### 1. MCP Protocol Adoption

The system uses the Model Context Protocol (MCP) instead of traditional REST APIs for several reasons:

- **Tool Discovery**: AI assistants can automatically discover available tools
- **Standardized Interface**: Consistent way for AI models to interact with services
- **Type Safety**: Built-in schema validation and type checking
- **Better Integration**: Works seamlessly with Claude Desktop, VS Code, and other MCP clients

### 2. Service Separation

The system separates the API Gateway from the Compliance Service to:

- **Isolate Concerns**: Each service has a clear, focused responsibility
- **Enable Scaling**: Services can be scaled independently
- **Facilitate Evolution**: Services can evolve at different rates
- **Support Multiple Clients**: Gateway provides a unified entry point

### 3. LangGraph for Workflow

LangGraph is used for workflow orchestration because:

- **State Management**: Maintains state throughout the analysis process
- **Flexible Flows**: Supports complex, multi-step analysis workflows
- **Traceability**: Provides visibility into the analysis process
- **Extensibility**: Easy to add new analysis steps

### 4. AI Integration Strategy

The system uses AI in multiple ways:

- **Orchestration**: Determines optimal processing strategy
- **Analysis**: Performs the core compliance analysis
- **Explanation**: Provides reasoning for decisions
- **Recommendations**: Generates actionable recommendations

## Technical Implementation

### MCP Tools

The system exposes several MCP tools:

- **analyze_compliance**: Comprehensive compliance analysis
- **quick_compliance_check**: Quick assessment of compliance
- **batch_analyze_expenses**: Process multiple expenses
- **get_employee_insights**: Employee-focused analytics
- **explain_compliance_decision**: Transparency into decisions

### MCP Resources

The system provides read-only data through resources:

- **health://gateway**: Gateway health status
- **health://service**: Service health status
- **capabilities://ai**: AI capabilities information
- **workflow://status**: Workflow status information

### MCP Prompts

The system offers pre-built prompts:

- **compliance_analysis_prompt**: For compliance analysis
- **quick_check_prompt**: For quick assessments
- **expense_analysis_prompt**: For detailed expense analysis

## Deployment Considerations

### Environment Requirements

- Python 3.12 or higher
- OpenAI API key
- Sufficient memory for LLM operations (4GB+ recommended)

### Scaling Strategy

- **Horizontal Scaling**: Deploy multiple instances of services
- **Vertical Scaling**: Increase resources for LLM operations
- **Caching**: Implement result caching for common queries

### Security Considerations

- **API Key Management**: Secure storage of OpenAI API keys
- **Data Privacy**: Minimize sensitive data exposure to LLMs
- **Input Validation**: Thorough validation of all inputs
- **Output Sanitization**: Ensure safe handling of LLM outputs

## Future Extensions

- **Additional Services**: Policy, pattern, recommendation, and escalation services
- **Custom LLM Integration**: Support for custom-trained compliance models
- **Advanced Caching**: Intelligent caching of analysis results
- **Feedback Loop**: Incorporate user feedback to improve AI analysis
- **Audit Trail**: Comprehensive logging for compliance auditing