#!/usr/bin/env python3

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import uuid

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.stdio import stdio_client
import httpx

# AI Integration imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create MCP server for the gateway
mcp = FastMCP("compliance-gateway")

# Configuration
AI_CONFIG = {
    "enabled": os.getenv("ENABLE_AI", "true").lower() == "true",
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "temperature": float(os.getenv("AI_TEMPERATURE", "0.1")),
}

# MCP Service connections configuration
MCP_SERVICES = {
    "compliance": {
        "command": "python",
        "args": ["./compliance_service.py"]
    },
    "policy": {
        "command": "python", 
        "args": ["./policy_service.py"]
    },
    "pattern": {
        "command": "python",
        "args": ["./pattern_analysis_service.py"]
    },
    "recommendation": {
        "command": "python",
        "args": ["./recommendation_service.py"]
    },
    "escalation": {
        "command": "python",
        "args": ["./escalation_service.py"]
    }
}

class IntelligentOrchestrator:
    """AI-powered service orchestration and decision making"""
    
    def __init__(self):
        self.llm_client = None
        self.mcp_clients = {}
        if AI_CONFIG["enabled"] and AI_CONFIG["openai_api_key"]:
            self._setup_ai_client()
    
    def _setup_ai_client(self):
        """Setup AI client for orchestration decisions"""
        try:
            self.llm_client = ChatOpenAI(
                openai_api_key=AI_CONFIG["openai_api_key"],
                model=AI_CONFIG["model"],
                temperature=AI_CONFIG["temperature"],
                max_tokens=1000
            )
            logger.info("AI orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to setup AI orchestrator: {e}")
            self.llm_client = None
    
    async def connect_to_mcp_service(self, service_name: str) -> Optional[ClientSession]:
        """Connect to an MCP service"""
        if service_name in self.mcp_clients:
            return self.mcp_clients[service_name]
        
        try:
            service_config = MCP_SERVICES.get(service_name)
            if not service_config:
                logger.error(f"No configuration found for service: {service_name}")
                return None
            
            async with stdio_client(service_config) as (read, write):
                session = ClientSession(read, write)
                await session.initialize()
                self.mcp_clients[service_name] = session
                logger.info(f"Connected to MCP service: {service_name}")
                return session
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP service {service_name}: {e}")
            return None
    
    async def call_mcp_tool(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool on an MCP service"""
        session = await self.connect_to_mcp_service(service_name)
        if not session:
            return None
        
        try:
            result = await session.call_tool(tool_name, arguments)
            return result.content if result else None
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on service {service_name}: {e}")
            return None
    
    async def determine_analysis_strategy(self, expense: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered decision on how to analyze the expense"""
        if not self.llm_client:
            return self._default_strategy(expense, options)
        
        strategy_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an intelligent compliance orchestrator. 
            Analyze the expense and determine the optimal analysis strategy.
            
            Consider:
            1. Expense amount and risk level
            2. Employee history and role
            3. Category and vendor
            4. Processing priority
            5. Resource optimization
            
            Return a strategy that balances thoroughness with efficiency."""),
            
            HumanMessage(content=f"""
            Expense Analysis Request:
            - Amount: ${expense.get('amount')}
            - Category: {expense.get('category')}
            - Vendor: {expense.get('vendor')}
            - Employee: {expense.get('employee_name')} ({expense.get('department')})
            - Priority: {options.get('priority', 'normal')}
            - Description: {expense.get('description')}
            
            Analysis Options Requested:
            {json.dumps(options.get('analysis_options', {}), indent=2)}
            
            Determine optimal strategy. Return JSON with:
            - use_ai_analysis: true/false
            - ai_analysis_depth: quick/standard/comprehensive
            - parallel_traditional: true/false (run traditional analysis in parallel)
            - services_to_call: list of services needed
            - estimated_processing_time: seconds
            - confidence_threshold: minimum confidence for auto-approval
            - reasoning: explanation of strategy choice
            """)
        ])
        
        try:
            response = await self.llm_client.ainvoke(strategy_prompt.format_messages())
            strategy = self._parse_ai_response(response.content)
            
            return {
                "use_ai_analysis": strategy.get("use_ai_analysis", True),
                "ai_analysis_depth": strategy.get("ai_analysis_depth", "standard"),
                "parallel_traditional": strategy.get("parallel_traditional", True),
                "services_to_call": strategy.get("services_to_call", ["compliance", "pattern", "recommendation"]),
                "estimated_processing_time": strategy.get("estimated_processing_time", 10),
                "confidence_threshold": strategy.get("confidence_threshold", 0.8),
                "reasoning": strategy.get("reasoning", "AI-determined optimal strategy"),
                "strategy_source": "ai"
            }
            
        except Exception as e:
            logger.error(f"AI strategy determination failed: {e}")
            return self._default_strategy(expense, options)
    
    def _default_strategy(self, expense: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy when AI is not available"""
        use_ai = options.get("analysis_options", {}).get("use_ai", True) and AI_CONFIG["enabled"]
        
        # Determine depth based on amount and priority
        if expense.get("amount", 0) > 500 or options.get("priority") in ["high", "urgent"]:
            depth = "comprehensive"
            services = ["compliance", "pattern", "recommendation", "escalation"]
        elif expense.get("amount", 0) > 100:
            depth = "standard"
            services = ["compliance", "pattern", "recommendation"]
        else:
            depth = "quick"
            services = ["compliance"]
        
        return {
            "use_ai_analysis": use_ai,
            "ai_analysis_depth": depth,
            "parallel_traditional": True,
            "services_to_call": services,
            "estimated_processing_time": 5,
            "confidence_threshold": 0.7,
            "reasoning": "Rule-based strategy determination",
            "strategy_source": "rules"
        }
    
    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response with fallback handling"""
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            return {}

# Initialize orchestrator
orchestrator = IntelligentOrchestrator()

# MCP Tools
@mcp.tool()
async def analyze_compliance(
    expense_id: str,
    employee_id: str,
    employee_name: str,
    amount: float,
    category: str,
    vendor: str,
    description: str,
    date: str,
    receipt_attached: bool,
    approval_status: str,
    department: str,
    analysis_options: Dict[str, bool] = None,
    priority: str = "normal",
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Perform intelligent compliance analysis with AI orchestration.
    
    Args:
        expense_id: Unique identifier for the expense
        employee_id: Employee identifier
        employee_name: Name of the employee
        amount: Expense amount
        category: Expense category
        vendor: Vendor name
        description: Expense description
        date: Date of expense
        receipt_attached: Whether receipt is attached
        approval_status: Current approval status
        department: Employee department
        analysis_options: Options for analysis (use_ai, full_analysis, etc.)
        priority: Processing priority (low, normal, high, urgent)
        context: Additional context for analysis
    
    Returns:
        Comprehensive compliance analysis result
    """
    start_time = time.time()
    analysis_id = f"analysis_{int(time.time())}_{expense_id}"
    
    logger.info(f"Starting compliance analysis {analysis_id}")
    
    expense_data = {
        "expense_id": expense_id,
        "employee_id": employee_id,
        "employee_name": employee_name,
        "amount": amount,
        "category": category,
        "vendor": vendor,
        "description": description,
        "date": date,
        "receipt_attached": receipt_attached,
        "approval_status": approval_status,
        "department": department,
        "metadata": context or {}
    }
    
    analysis_options = analysis_options or {
        "use_ai": True,
        "full_ai_analysis": True,
        "check_policy": True,
        "check_patterns": True,
        "generate_recommendations": True,
        "assess_escalation": True
    }
    
    # Step 1: Determine analysis strategy
    strategy = await orchestrator.determine_analysis_strategy(
        expense_data, 
        {"analysis_options": analysis_options, "priority": priority}
    )
    logger.info(f"Analysis strategy: {strategy['reasoning']}")
    
    # Step 2: Execute analysis based on strategy
    results = {
        "expense_id": expense_id,
        "analysis_id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "ai_enhanced": False,
        "processing_metadata": {
            "strategy": strategy,
            "start_time": start_time,
            "services_called": []
        }
    }
    
    if strategy["use_ai_analysis"]:
        # Call AI compliance service
        ai_result = await orchestrator.call_mcp_tool(
            "compliance",
            "analyze_compliance_with_ai",
            {
                **expense_data,
                "full_analysis": strategy["ai_analysis_depth"] == "comprehensive"
            }
        )
        
        if ai_result:
            results["ai_analysis"] = ai_result
            results["ai_enhanced"] = True
            results["processing_metadata"]["ai_enhanced"] = True
    
    # Call other services based on strategy
    for service_name in strategy.get("services_to_call", []):
        if service_name == "compliance":
            continue  # Already handled above
        
        service_result = await orchestrator.call_mcp_tool(
            service_name,
            f"analyze_{service_name}",
            expense_data
        )
        
        if service_result:
            results[f"{service_name}_analysis"] = service_result
            results["processing_metadata"]["services_called"].append(service_name)
    
    # Finalize metadata
    results["processing_metadata"].update({
        "end_time": time.time(),
        "total_processing_time": time.time() - start_time,
        "analysis_completed": True
    })
    
    logger.info(f"Compliance analysis {analysis_id} completed in {results['processing_metadata']['total_processing_time']:.2f}s")
    return results

@mcp.tool()
async def quick_compliance_check(
    expense_id: str,
    amount: float,
    category: str,
    vendor: str,
    receipt_attached: bool
) -> Dict[str, Any]:
    """Perform quick compliance check for low-risk expenses.
    
    Args:
        expense_id: Unique identifier for the expense
        amount: Expense amount
        category: Expense category
        vendor: Vendor name
        receipt_attached: Whether receipt is attached
    
    Returns:
        Quick assessment result
    """
    try:
        # Try AI quick assessment first
        if AI_CONFIG["enabled"]:
            ai_result = await orchestrator.call_mcp_tool(
                "compliance",
                "quick_ai_assessment",
                {
                    "expense_id": expense_id,
                    "amount": amount,
                    "category": category,
                    "vendor": vendor,
                    "receipt_attached": receipt_attached
                }
            )
            
            if ai_result:
                return {
                    "expense_id": expense_id,
                    "quick_assessment": "ai_powered",
                    "risk_level": ai_result.get("risk_level", "MEDIUM"),
                    "compliance_score": ai_result.get("compliance_score", 50),
                    "approval_recommendation": "APPROVE" if ai_result.get("compliance_score", 0) > 80 else "REVIEW",
                    "processing_time": "< 2 seconds",
                    "ai_enhanced": True
                }
        
        # Fallback to simple rule-based check
        risk_score = 0
        if amount > 100:
            risk_score += 20
        if not receipt_attached:
            risk_score += 30
        if category in ["entertainment", "alcohol"]:
            risk_score += 25
        
        return {
            "expense_id": expense_id,
            "quick_assessment": "rule_based",
            "risk_level": "HIGH" if risk_score > 50 else "MEDIUM" if risk_score > 25 else "LOW",
            "compliance_score": max(0, 100 - risk_score),
            "approval_recommendation": "APPROVE" if risk_score < 25 else "REVIEW",
            "processing_time": "< 1 second",
            "ai_enhanced": False
        }
        
    except Exception as e:
        logger.error(f"Quick compliance check failed: {e}")
        return {"error": str(e)}

@mcp.tool()
async def batch_analyze_expenses(
    expenses: List[Dict[str, Any]],
    batch_options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process multiple expenses with intelligent prioritization.
    
    Args:
        expenses: List of expense records to analyze
        batch_options: Options for batch processing
    
    Returns:
        Batch analysis results
    """
    try:
        batch_options = batch_options or {}
        batch_id = f"batch_{int(time.time())}"
        
        # AI-powered batch prioritization
        if AI_CONFIG["enabled"] and len(expenses) > 5:
            prioritized_expenses = await _prioritize_batch_with_ai(expenses)
        else:
            prioritized_expenses = sorted(expenses, key=lambda x: x.get("amount", 0), reverse=True)
        
        results = []
        for i, expense in enumerate(prioritized_expenses):
            try:
                # Use quick check for low-risk items in large batches
                if len(expenses) > 10 and expense.get("amount", 0) < 50:
                    result = await quick_compliance_check(
                        expense_id=expense["expense_id"],
                        amount=expense["amount"],
                        category=expense["category"],
                        vendor=expense["vendor"],
                        receipt_attached=expense.get("receipt_attached", False)
                    )
                    result["batch_position"] = i + 1
                    result["processing_method"] = "quick_check"
                else:
                    # Full analysis for higher risk items
                    result = await analyze_compliance(**expense)
                    result["batch_position"] = i + 1
                    result["processing_method"] = "full_analysis"
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch item {expense.get('expense_id')} failed: {e}")
                results.append({
                    "expense_id": expense.get("expense_id"),
                    "error": str(e),
                    "batch_position": i + 1
                })
        
        return {
            "batch_id": batch_id,
            "total_expenses": len(expenses),
            "processed_count": len(results),
            "results": results,
            "batch_summary": {
                "high_risk_count": sum(1 for r in results if r.get("risk_level") == "HIGH"),
                "auto_approved_count": sum(1 for r in results if r.get("approval_recommendation") == "APPROVE"),
                "requires_review_count": sum(1 for r in results if r.get("approval_recommendation") in ["REVIEW", "REVIEW_REQUIRED"]),
                "ai_enhanced_count": sum(1 for r in results if r.get("ai_enhanced", False))
            },
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return {"error": str(e)}

async def _prioritize_batch_with_ai(expenses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use AI to prioritize batch processing order"""
    try:
        if not AI_CONFIG["enabled"]:
            return expenses
        
        # Call AI service for batch prioritization
        batch_result = await orchestrator.call_mcp_tool(
            "compliance",
            "batch_analyze_compliance",
            {
                "expenses": expenses,
                "full_analysis": False
            }
        )
        
        if batch_result and "results" in batch_result:
            # Sort by risk level from AI analysis
            risk_priorities = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            
            results_with_priority = []
            for result in batch_result.get("results", []):
                expense_id = result.get("expense_id")
                risk_level = result.get("quick_assessment", {}).get("risk_level", "MEDIUM")
                priority = risk_priorities.get(risk_level, 2)
                
                # Find corresponding expense
                expense = next((exp for exp in expenses if exp.get("expense_id") == expense_id), None)
                if expense:
                    results_with_priority.append((priority, expense))
            
            # Sort by priority (highest first)
            results_with_priority.sort(key=lambda x: x[0], reverse=True)
            return [expense for priority, expense in results_with_priority]
                
    except Exception as e:
        logger.error(f"AI batch prioritization failed: {e}")
    
    # Fallback to amount-based sorting
    return sorted(expenses, key=lambda x: x.get("amount", 0), reverse=True)

# MCP Resources
@mcp.resource("health://gateway")
async def gateway_health() -> str:
    """Get comprehensive health status of the gateway and connected services"""
    health_results = {
        "gateway_status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "ai_enhanced": True,
        "services": {},
        "ai_capabilities": {}
    }
    
    # Check MCP services
    for service_name in MCP_SERVICES.keys():
        try:
            # Try to connect and check health
            session = await orchestrator.connect_to_mcp_service(service_name)
            if session:
                health_results["services"][service_name] = {
                    "status": "healthy",
                    "connected": True
                }
            else:
                health_results["services"][service_name] = {
                    "status": "unavailable",
                    "connected": False
                }
        except Exception as e:
            health_results["services"][service_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Check AI capabilities
    if AI_CONFIG["enabled"]:
        health_results["ai_capabilities"] = {
            "intelligent_orchestration": orchestrator.llm_client is not None,
            "batch_prioritization": True,
            "decision_explanation": True
        }
    
    return json.dumps(health_results, indent=2)

@mcp.resource("capabilities://ai")
async def ai_capabilities() -> str:
    """Get detailed AI capabilities and status"""
    capabilities = {
        "ai_enabled": AI_CONFIG["enabled"],
        "ai_model": AI_CONFIG["model"],
        "capabilities": {
            "intelligent_orchestration": orchestrator.llm_client is not None,
            "full_ai_analysis": True,
            "quick_ai_assessment": True,
            "batch_prioritization": True,
            "decision_explanation": True,
            "mcp_protocol_support": True,
        },
        "processing_modes": {
            "quick_check": "< 2 seconds",
            "standard_analysis": "5-10 seconds", 
            "comprehensive_analysis": "10-30 seconds",
            "batch_processing": "Variable based on size"
        },
        "ai_insights_available": [
            "policy_reasoning",
            "pattern_detection", 
            "risk_assessment",
            "contextual_recommendations",
            "escalation_decisions"
        ],
        "status_checked_at": datetime.now().isoformat()
    }
    return json.dumps(capabilities, indent=2)

# MCP Prompts
@mcp.prompt()
def expense_analysis_prompt(
    amount: float,
    category: str,
    vendor: str,
    department: str
) -> str:
    """Generate a prompt for expense analysis"""
    return f"""Please analyze this expense for compliance:
    
    Amount: ${amount}
    Category: {category}
    Vendor: {vendor}
    Department: {department}
    
    Use the analyze_compliance tool to perform a comprehensive analysis."""

@mcp.prompt()
def batch_analysis_prompt(expense_count: int) -> str:
    """Generate a prompt for batch expense analysis"""
    return f"""I need to analyze {expense_count} expenses for compliance.
    
    Please use the batch_analyze_expenses tool to process them efficiently,
    prioritizing high-risk expenses first."""

@mcp.prompt()
def quick_check_prompt(expense_id: str) -> str:
    """Generate a prompt for quick compliance check"""
    return f"""Please perform a quick compliance check on expense {expense_id}.
    
    Use the quick_compliance_check tool for rapid assessment."""

# Lifecycle management
async def on_startup():
    """Initialize services on startup"""
    logger.info("MCP Compliance Gateway starting up...")
    
    if AI_CONFIG["enabled"] and AI_CONFIG["openai_api_key"]:
        logger.info("AI features enabled")
    else:
        logger.warning("AI features disabled or API key not configured")
    
    logger.info("MCP Compliance Gateway ready")

async def on_shutdown():
    """Cleanup on shutdown"""
    logger.info("MCP Compliance Gateway shutting down...")
    
    # Close all MCP client connections
    for service_name, session in orchestrator.mcp_clients.items():
        try:
            await session.close()
            logger.info(f"Closed connection to {service_name}")
        except Exception as e:
            logger.error(f"Error closing connection to {service_name}: {e}")
    
    logger.info("MCP Compliance Gateway shutdown complete")

if __name__ == "__main__":
    # Run startup function before starting the server
    asyncio.run(on_startup())
    
    # Run the MCP server
    try:
        mcp.run()
    finally:
        # Run shutdown function when server stops
        asyncio.run(on_shutdown())