#!/usr/bin/env python3

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Annotated
from enum import Enum

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# AI and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("compliance-service")

# Configuration
AI_CONFIG = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "temperature": float(os.getenv("AI_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("AI_MAX_TOKENS", "4000")),
}

# Enums
class ComplianceRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AIAnalysisType(str, Enum):
    POLICY_REASONING = "policy_reasoning"
    PATTERN_DETECTION = "pattern_detection"
    RISK_ASSESSMENT = "risk_assessment"
    RECOMMENDATION = "recommendation"
    ESCALATION_DECISION = "escalation_decision"

# LangGraph State Definition
class ComplianceAnalysisState(TypedDict):
    expense: Dict[str, Any]
    employee_context: Dict[str, Any]
    policy_rules: Dict[str, Any]
    historical_data: Dict[str, Any]
    current_analysis: Dict[str, Any]
    violations: List[Dict[str, Any]]
    ai_insights: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    escalation_decision: Dict[str, Any]
    messages: Annotated[List, add_messages]
    final_result: Optional[Dict[str, Any]]

class AIComplianceService:
    def __init__(self):
        self.llm_client = None
        self.compliance_graph = None
        
        if os.getenv("OPENAI_API_KEY"):
            self._setup_ai_client()
            self._setup_compliance_graph()
        else:
            logger.warning("OpenAI API key not found. AI features will be limited.")
    
    def _setup_ai_client(self):
        """Setup AI client"""
        try:
            self.llm_client = ChatOpenAI(
                openai_api_key=AI_CONFIG["openai_api_key"],
                model=AI_CONFIG["model"],
                temperature=AI_CONFIG["temperature"],
                max_tokens=AI_CONFIG["max_tokens"]
            )
            logger.info(f"AI client initialized with model: {AI_CONFIG['model']}")
        except Exception as e:
            logger.error(f"Failed to setup AI client: {e}")
            self.llm_client = None
    
    def _setup_compliance_graph(self):
        """Setup LangGraph workflow for compliance analysis"""
        if not self.llm_client:
            return
        
        # Create the workflow graph
        workflow = StateGraph(ComplianceAnalysisState)
        
        # Add nodes
        workflow.add_node("gather_context", self._gather_context_node)
        workflow.add_node("policy_analysis", self._policy_analysis_node)
        workflow.add_node("pattern_detection", self._pattern_detection_node)
        # Change the node name to avoid conflict with state key
        workflow.add_node("risk_assessment_node", self._risk_assessment_node)
        workflow.add_node("generate_recommendations", self._recommendations_node)
        # Change the node name to avoid conflict with state key
        workflow.add_node("escalation_decision_node", self._escalation_decision_node)
        workflow.add_node("finalize_result", self._finalize_result_node)
        
        # Define the workflow
        workflow.set_entry_point("gather_context")
        workflow.add_edge("gather_context", "policy_analysis")
        workflow.add_edge("policy_analysis", "pattern_detection")
        # Update the edge connections to use the new node names
        workflow.add_edge("pattern_detection", "risk_assessment_node")
        workflow.add_edge("risk_assessment_node", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "escalation_decision_node")
        workflow.add_edge("escalation_decision_node", "finalize_result")
        workflow.add_edge("finalize_result", END)
        
        self.compliance_graph = workflow.compile()
        logger.info("LangGraph compliance workflow initialized")
    
    async def _gather_context_node(self, state: ComplianceAnalysisState) -> Dict[str, Any]:
        """Gather context for compliance analysis"""
        expense = state["expense"]
        
        # In a real implementation, these would fetch from databases or other MCP services
        # For now, we'll create mock data
        employee_context = {
            "employee_id": expense.get("employee_id"),
            "role": "Senior Analyst",
            "tenure_years": 3,
            "previous_violations": 0,
            "spending_limit": 1000
        }
        
        policy_rules = {
            "max_amount_by_category": {
                "meals": 50,
                "travel": 500,
                "entertainment": 100,
                "supplies": 200
            },
            "receipt_required_above": 25,
            "pre_approval_required_above": 500
        }
        
        historical_data = {
            "avg_monthly_spend": 450,
            "expense_count_last_90_days": 12,
            "common_categories": ["meals", "travel", "supplies"]
        }
        
        return {
            "employee_context": employee_context,
            "policy_rules": policy_rules,
            "historical_data": historical_data,
            "messages": [SystemMessage(content="Beginning comprehensive compliance analysis...")]
        }
    
    async def _policy_analysis_node(self, state: ComplianceAnalysisState) -> Dict[str, Any]:
        """AI-powered policy compliance analysis"""
        if not self.llm_client:
            return {"violations": [], "ai_insights": []}
        
        expense = state["expense"]
        policy_rules = state.get("policy_rules", {})
        employee_context = state.get("employee_context", {})
        
        # Create policy analysis prompt
        policy_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert compliance analyst with deep knowledge of corporate expense policies. 
            Analyze the given expense against provided policies and identify any violations with detailed reasoning.
            
            Focus on:
            1. Amount limits and thresholds
            2. Category restrictions
            3. Vendor approval requirements
            4. Receipt and documentation requirements
            5. Employee-specific limits
            6. Department-specific rules
            
            Provide detailed reasoning for each violation and assess confidence levels."""),
            
            HumanMessage(content=f"""
            Expense Details:
            - Amount: ${expense.get('amount')}
            - Category: {expense.get('category')}
            - Vendor: {expense.get('vendor')}
            - Description: {expense.get('description')}
            - Receipt Attached: {expense.get('receipt_attached')}
            - Department: {expense.get('department')}
            - Employee: {expense.get('employee_name')} ({expense.get('employee_id')})
            
            Policy Rules:
            {json.dumps(policy_rules, indent=2)}
            
            Employee Context:
            {json.dumps(employee_context, indent=2)}
            
            Analyze this expense for policy violations and provide detailed reasoning.
            Return a JSON response with violations array, each containing:
            - violation_type
            - severity (LOW/MEDIUM/HIGH/CRITICAL)
            - description
            - reasoning
            - confidence_score (0-1)
            - evidence
            - policy_references
            - recommended_actions
            """)
        ])
        
        try:
            response = await self.llm_client.ainvoke(policy_prompt.format_messages())
            
            # Parse AI response
            violations_data = self._parse_ai_response(response.content)
            violations = []
            
            for violation_data in violations_data.get("violations", []):
                violation = {
                    "violation_id": f"viol_{uuid.uuid4().hex[:8]}",
                    "type": violation_data.get("violation_type", "unknown"),
                    "severity": violation_data.get("severity", "MEDIUM"),
                    "description": violation_data.get("description", ""),
                    "ai_reasoning": violation_data.get("reasoning", ""),
                    "confidence_score": violation_data.get("confidence_score", 0.5),
                    "evidence": violation_data.get("evidence", []),
                    "policy_references": violation_data.get("policy_references", []),
                    "recommended_actions": violation_data.get("recommended_actions", [])
                }
                violations.append(violation)
            
            # Create AI insight
            insight = {
                "insight_id": f"insight_{uuid.uuid4().hex[:8]}",
                "type": AIAnalysisType.POLICY_REASONING.value,
                "confidence": 0.9,
                "reasoning": "AI-powered policy analysis using Claude",
                "evidence": [f"Analyzed {len(policy_rules)} policy rules"],
                "recommendations": [f"Found {len(violations)} potential violations"],
                "risk_factors": [v["type"] for v in violations],
                "generated_at": datetime.now().isoformat(),
                "model_used": AI_CONFIG["model"]
            }
            
            return {
                "violations": violations,
                "ai_insights": [insight],
                "policy_analysis": {
                    "analysis_completed": True,
                    "violations_found": len(violations),
                    "ai_reasoning": response.content[:500] + "..." if len(response.content) > 500 else response.content
                },
                "messages": [AIMessage(content=f"Policy analysis complete. Found {len(violations)} violations.")]
            }
            
        except Exception as e:
            logger.error(f"Policy analysis failed: {e}")
            return {
                "violations": [],
                "ai_insights": [],
                "policy_analysis": {"error": str(e)},
                "messages": [AIMessage(content="Policy analysis failed")]
            }
    
    async def _pattern_detection_node(self, state: ComplianceAnalysisState) -> Dict[str, Any]:
        """AI-powered pattern detection and anomaly analysis"""
        if not self.llm_client:
            return {"pattern_analysis": {}}
        
        expense = state["expense"]
        historical_data = state.get("historical_data", {})
        
        pattern_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in expense pattern analysis and anomaly detection.
            Analyze the current expense against historical patterns to identify anomalies, trends, and concerning behaviors.
            
            Focus on:
            1. Amount anomalies (unusually high/low)
            2. Frequency patterns
            3. Vendor selection patterns
            4. Category distribution changes
            5. Timing anomalies
            6. Behavioral shifts
            
            Provide detailed analysis with confidence scores and specific recommendations."""),
            
            HumanMessage(content=f"""
            Current Expense:
            - Amount: ${expense.get('amount')}
            - Category: {expense.get('category')}
            - Vendor: {expense.get('vendor')}
            - Date: {expense.get('date')}
            - Description: {expense.get('description')}
            
            Historical Data (Last 90 days):
            {json.dumps(historical_data, indent=2)}
            
            Analyze patterns and detect anomalies. Return JSON with:
            - anomalies: list of detected anomalies
            - patterns: identified spending patterns
            - risk_indicators: behavioral risk factors
            - confidence_scores: for each analysis
            - recommendations: pattern-based suggestions
            """)
        ])
        
        try:
            response = await self.llm_client.ainvoke(pattern_prompt.format_messages())
            pattern_data = self._parse_ai_response(response.content)
            
            # Create pattern analysis insight
            insight = {
                "insight_id": f"insight_{uuid.uuid4().hex[:8]}",
                "type": AIAnalysisType.PATTERN_DETECTION.value,
                "confidence": 0.85,
                "reasoning": "AI-powered pattern analysis using historical data",
                "evidence": [f"Analyzed historical spending patterns"],
                "recommendations": pattern_data.get("recommendations", []),
                "risk_factors": pattern_data.get("risk_indicators", []),
                "generated_at": datetime.now().isoformat(),
                "model_used": AI_CONFIG["model"]
            }
            
            current_insights = state.get("ai_insights", [])
            current_insights.append(insight)
            
            return {
                "ai_insights": current_insights,
                "pattern_analysis": {
                    "anomalies": pattern_data.get("anomalies", []),
                    "patterns": pattern_data.get("patterns", {}),
                    "risk_indicators": pattern_data.get("risk_indicators", []),
                    "ai_analysis": response.content[:500] + "..." if len(response.content) > 500 else response.content
                },
                "messages": [AIMessage(content="Pattern analysis complete with AI insights.")]
            }
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {
                "pattern_analysis": {"error": str(e)},
                "messages": [AIMessage(content="Pattern analysis encountered an error")]
            }
    
    async def _risk_assessment_node(self, state: ComplianceAnalysisState) -> Dict[str, Any]:
        """AI-powered comprehensive risk assessment"""
        if not self.llm_client:
            return {"risk_assessment": {}}
        
        violations = state.get("violations", [])
        pattern_analysis = state.get("pattern_analysis", {})
        employee_context = state.get("employee_context", {})
        
        risk_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a senior risk assessment specialist for corporate compliance.
            Evaluate the overall compliance risk based on violations, patterns, and employee context.
            
            Consider:
            1. Severity and frequency of violations
            2. Pattern anomalies and trends
            3. Employee history and role
            4. Financial impact
            5. Regulatory implications
            6. Reputational risks
            
            Provide a comprehensive risk assessment with scoring and mitigation strategies."""),
            
            HumanMessage(content=f"""
            Violations Found:
            {json.dumps([{
                'type': v['type'],
                'severity': v['severity'],
                'confidence': v['confidence_score'],
                'description': v['description']
            } for v in violations], indent=2)}
            
            Pattern Analysis:
            {json.dumps(pattern_analysis, indent=2)}
            
            Employee Context:
            {json.dumps(employee_context, indent=2)}
            
            Provide comprehensive risk assessment. Return JSON with:
            - overall_risk_level: LOW/MEDIUM/HIGH/CRITICAL
            - risk_score: 0-100
            - risk_factors: detailed list
            - impact_assessment: potential consequences
            - mitigation_strategies: recommended actions
            - confidence_level: assessment confidence
            """)
        ])
        
        try:
            response = await self.llm_client.ainvoke(risk_prompt.format_messages())
            risk_data = self._parse_ai_response(response.content)
            
            # Create risk assessment insight
            insight = {
                "insight_id": f"insight_{uuid.uuid4().hex[:8]}",
                "type": AIAnalysisType.RISK_ASSESSMENT.value,
                "confidence": risk_data.get("confidence_level", 0.8),
                "reasoning": "Comprehensive AI risk assessment",
                "evidence": [f"Evaluated {len(violations)} violations and pattern analysis"],
                "recommendations": risk_data.get("mitigation_strategies", []),
                "risk_factors": risk_data.get("risk_factors", []),
                "generated_at": datetime.now().isoformat(),
                "model_used": AI_CONFIG["model"]
            }
            
            current_insights = state.get("ai_insights", [])
            current_insights.append(insight)
            
            return {
                "ai_insights": current_insights,
                "risk_assessment": {
                    "overall_risk_level": risk_data.get("overall_risk_level", "MEDIUM"),
                    "risk_score": risk_data.get("risk_score", 50),
                    "risk_factors": risk_data.get("risk_factors", []),
                    "impact_assessment": risk_data.get("impact_assessment", {}),
                    "mitigation_strategies": risk_data.get("mitigation_strategies", []),
                    "ai_reasoning": response.content[:500] + "..." if len(response.content) > 500 else response.content
                },
                "messages": [AIMessage(content=f"Risk assessment complete. Risk level: {risk_data.get('overall_risk_level', 'MEDIUM')}")]
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                "risk_assessment": {"error": str(e)},
                "messages": [AIMessage(content="Risk assessment encountered an error")]
            }
    
    async def _recommendations_node(self, state: ComplianceAnalysisState) -> Dict[str, Any]:
        """AI-powered recommendation generation"""
        if not self.llm_client:
            return {"recommendations": {}}
        
        violations = state.get("violations", [])
        risk_assessment = state.get("risk_assessment", {})
        pattern_analysis = state.get("pattern_analysis", {})
        employee_context = state.get("employee_context", {})
        
        recommendations_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert compliance advisor specializing in actionable recommendations.
            Generate specific, prioritized recommendations based on the complete analysis.
            
            Provide:
            1. Immediate actions required
            2. Short-term improvements
            3. Long-term preventive measures
            4. Training and education needs
            5. Process improvements
            6. Policy updates if needed
            
            Each recommendation should be specific, actionable, and include implementation guidance."""),
            
            HumanMessage(content=f"""
            Analysis Summary:
            - Violations: {len(violations)}
            - Risk Level: {risk_assessment.get('overall_risk_level', 'UNKNOWN')}
            - Risk Score: {risk_assessment.get('risk_score', 0)}/100
            
            Detailed Analysis:
            Violations: {json.dumps(violations, indent=2)}
            Risk Assessment: {json.dumps(risk_assessment, indent=2)}
            Pattern Analysis: {json.dumps(pattern_analysis, indent=2)}
            Employee Context: {json.dumps(employee_context, indent=2)}
            
            Generate comprehensive recommendations. Return JSON with:
            - immediate_actions: urgent steps
            - short_term_recommendations: 1-4 weeks
            - long_term_strategies: 1-6 months
            - training_needs: educational requirements
            - process_improvements: workflow enhancements
            - policy_updates: suggested policy changes
            - priority_scores: 1-10 for each recommendation
            """)
        ])
        
        try:
            response = await self.llm_client.ainvoke(recommendations_prompt.format_messages())
            recommendations_data = self._parse_ai_response(response.content)
            
            # Create recommendations insight
            insight = {
                "insight_id": f"insight_{uuid.uuid4().hex[:8]}",
                "type": AIAnalysisType.RECOMMENDATION.value,
                "confidence": 0.9,
                "reasoning": "AI-generated actionable recommendations",
                "evidence": ["Comprehensive analysis of violations, risks, and patterns"],
                "recommendations": recommendations_data.get("immediate_actions", []),
                "risk_factors": [],
                "generated_at": datetime.now().isoformat(),
                "model_used": AI_CONFIG["model"]
            }
            
            current_insights = state.get("ai_insights", [])
            current_insights.append(insight)
            
            return {
                "ai_insights": current_insights,
                "recommendations": recommendations_data,
                "messages": [AIMessage(content="Comprehensive recommendations generated")]
            }
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return {
                "recommendations": {"error": str(e)},
                "messages": [AIMessage(content="Recommendations generation encountered an error")]
            }
    
    async def _escalation_decision_node(self, state: ComplianceAnalysisState) -> Dict[str, Any]:
        """AI-powered escalation decision making"""
        if not self.llm_client:
            return {"escalation_decision": {}}
        
        violations = state.get("violations", [])
        risk_assessment = state.get("risk_assessment", {})
        employee_context = state.get("employee_context", {})
        
        escalation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a senior compliance officer making escalation decisions.
            Determine if and how this expense should be escalated based on violations, risk, and context.
            
            Consider:
            1. Violation severity and frequency
            2. Risk level and potential impact
            3. Employee history and role
            4. Regulatory requirements
            5. Company policies
            6. Cost-benefit of escalation
            
            Make clear, justified decisions with specific escalation paths."""),
            
            HumanMessage(content=f"""
            Escalation Decision Required:
            
            Violations: {json.dumps([{
                'type': v['type'],
                'severity': v['severity'],
                'confidence': v['confidence_score']
            } for v in violations], indent=2)}
            
            Risk Assessment: {json.dumps(risk_assessment, indent=2)}
            Employee Context: {json.dumps(employee_context, indent=2)}
            
            Make escalation decision. Return JSON with:
            - escalate: true/false
            - escalation_level: SUPERVISOR/MANAGER/DIRECTOR/EXECUTIVE/COMPLIANCE
            - urgency: LOW/MEDIUM/HIGH/CRITICAL
            - recipients: who should be notified
            - escalation_reason: detailed justification
            - timeline: when escalation should occur
            - required_actions: what must be done
            - auto_approve: whether expense can be auto-approved despite issues
            """)
        ])
        
        try:
            response = await self.llm_client.ainvoke(escalation_prompt.format_messages())
            escalation_data = self._parse_ai_response(response.content)
            
            # Create escalation insight
            insight = {
                "insight_id": f"insight_{uuid.uuid4().hex[:8]}",
                "type": AIAnalysisType.ESCALATION_DECISION.value,
                "confidence": 0.85,
                "reasoning": "AI-powered escalation decision",
                "evidence": ["Analysis of violations and risk factors"],
                "recommendations": escalation_data.get("required_actions", []),
                "risk_factors": [],
                "generated_at": datetime.now().isoformat(),
                "model_used": AI_CONFIG["model"]
            }
            
            current_insights = state.get("ai_insights", [])
            current_insights.append(insight)
            
            return {
                "ai_insights": current_insights,
                "escalation_decision": escalation_data,
                "messages": [AIMessage(content=f"Escalation decision: {'Escalate' if escalation_data.get('escalate') else 'No escalation required'}")]
            }
            
        except Exception as e:
            logger.error(f"Escalation decision failed: {e}")
            return {
                "escalation_decision": {"error": str(e)},
                "messages": [AIMessage(content="Escalation decision encountered an error")]
            }
    
    async def _finalize_result_node(self, state: ComplianceAnalysisState) -> Dict[str, Any]:
        """Finalize comprehensive compliance analysis result"""
        expense = state["expense"]
        violations = state.get("violations", [])
        ai_insights = state.get("ai_insights", [])
        risk_assessment = state.get("risk_assessment", {})
        
        # Calculate overall compliance score
        risk_score = risk_assessment.get("risk_score", 50)
        compliance_score = max(0, 100 - risk_score)
        
        # Determine overall risk level
        overall_risk_level = ComplianceRiskLevel.MEDIUM.value
        if risk_score >= 80:
            overall_risk_level = ComplianceRiskLevel.CRITICAL.value
        elif risk_score >= 60:
            overall_risk_level = ComplianceRiskLevel.HIGH.value
        elif risk_score >= 30:
            overall_risk_level = ComplianceRiskLevel.MEDIUM.value
        else:
            overall_risk_level = ComplianceRiskLevel.LOW.value
        
        # Create final result
        final_result = {
            "expense_id": expense.get("expense_id"),
            "employee_id": expense.get("employee_id"),
            "overall_risk_level": overall_risk_level,
            "compliance_score": compliance_score,
            "violations": violations,
            "ai_insights": ai_insights,
            "policy_analysis": state.get("policy_analysis", {}),
            "pattern_analysis": state.get("pattern_analysis", {}),
            "recommendations": state.get("recommendations", {}),
            "escalation_recommendation": state.get("escalation_decision", {}),
            "processing_metadata": {
                "analysis_completed_at": datetime.now().isoformat(),
                "ai_model_used": AI_CONFIG["model"],
                "workflow_version": "3.0.0",
                "total_insights_generated": len(ai_insights),
            },
        }
        
        return {
            "final_result": final_result,
            "messages": [AIMessage(content="Comprehensive AI compliance analysis completed")]
        }
    
    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response, handling potential JSON extraction"""
        try:
            # Try to parse as direct JSON
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Extract JSON from markdown code blocks or other formats
            import re
            
            # Look for JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Look for JSON-like content
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Fallback to empty structure
            logger.warning("Could not parse AI response as JSON")
            return {}
    
    async def analyze_compliance_with_ai(self, expense: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive AI-powered compliance analysis using LangGraph workflow"""
        if not self.compliance_graph:
            raise Exception("AI compliance analysis not available")
        
        logger.info(f"Starting AI compliance analysis for expense {expense.get('expense_id')}")
        
        # Initialize state
        initial_state = ComplianceAnalysisState(
            expense=expense,
            employee_context={},
            policy_rules={},
            historical_data={},
            current_analysis={},
            violations=[],
            ai_insights=[],
            risk_assessment={},
            recommendations=[],
            escalation_decision={},
            messages=[],
            final_result=None
        )
        
        try:
            # Run the LangGraph workflow
            final_state = await self.compliance_graph.ainvoke(initial_state)
            
            result = final_state.get("final_result")
            if result:
                logger.info(f"AI compliance analysis completed for {expense.get('expense_id')}")
                return result
            else:
                raise Exception("Workflow completed but no final result generated")
                
        except Exception as e:
            logger.error(f"AI compliance analysis failed for {expense.get('expense_id')}: {e}")
            
            # Return fallback result
            return {
                "expense_id": expense.get("expense_id"),
                "employee_id": expense.get("employee_id"),
                "overall_risk_level": ComplianceRiskLevel.MEDIUM.value,
                "compliance_score": 50.0,
                "violations": [],
                "ai_insights": [],
                "policy_analysis": {"error": str(e)},
                "pattern_analysis": {"error": str(e)},
                "recommendations": {"error": str(e)},
                "escalation_recommendation": {"error": str(e)},
                "processing_metadata": {
                    "error": str(e),
                    "analysis_completed_at": datetime.now().isoformat(),
                    "workflow_version": "3.0.0"
                },
            }
    
    async def quick_ai_assessment(self, expense: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Quick AI assessment without full workflow for simple cases"""
        if not self.llm_client:
            return {"error": "AI client not available"}
        
        context = context or {}
        
        quick_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a compliance AI assistant. Provide a quick assessment of this expense.
            Focus on obvious violations and immediate concerns. Be concise but thorough."""),
            
            HumanMessage(content=f"""
            Quick Assessment Needed:
            
            Expense: ${expense.get('amount')} for {expense.get('category')} at {expense.get('vendor')}
            Description: {expense.get('description')}
            Receipt: {'Yes' if expense.get('receipt_attached') else 'No'}
            Employee: {expense.get('employee_name')} from {expense.get('department')}
            
            Context: {json.dumps(context, indent=2)}
            
            Provide quick assessment with:
            - risk_level: LOW/MEDIUM/HIGH/CRITICAL
            - key_concerns: list of main issues
            - immediate_actions: what to do now
            - requires_full_analysis: true/false
            """)
        ])
        
        try:
            response = await self.llm_client.ainvoke(quick_prompt.format_messages())
            return self._parse_ai_response(response.content)
        except Exception as e:
            logger.error(f"Quick AI assessment failed: {e}")
            return {"error": str(e)}

# Initialize AI compliance service
ai_compliance_service = AIComplianceService()

# MCP Tools
@mcp.tool()
async def analyze_compliance_with_ai(
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
    metadata: Dict[str, Any] = None,
    full_analysis: bool = True
) -> Dict[str, Any]:
    """Comprehensive AI-powered compliance analysis using LangGraph.
    
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
        metadata: Additional metadata
        full_analysis: Whether to run full analysis or quick assessment
        
    Returns:
        Comprehensive compliance analysis result
    """
    try:
        expense = {
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
            "metadata": metadata or {}
        }
        
        if full_analysis:
            result = await ai_compliance_service.analyze_compliance_with_ai(expense)
            return result
        else:
            # Quick assessment
            quick_result = await ai_compliance_service.quick_ai_assessment(expense)
            
            # Convert to full result format
            risk_level = ComplianceRiskLevel.MEDIUM.value
            if quick_result.get("risk_level") == "LOW":
                risk_level = ComplianceRiskLevel.LOW.value
            elif quick_result.get("risk_level") == "HIGH":
                risk_level = ComplianceRiskLevel.HIGH.value
            elif quick_result.get("risk_level") == "CRITICAL":
                risk_level = ComplianceRiskLevel.CRITICAL.value
            
            return {
                "expense_id": expense_id,
                "employee_id": employee_id,
                "overall_risk_level": risk_level,
                "compliance_score": 75.0 if risk_level == ComplianceRiskLevel.LOW.value else 50.0,
                "violations": [],
                "ai_insights": [],
                "policy_analysis": {"quick_assessment": quick_result},
                "pattern_analysis": {},
                "recommendations": {"immediate_actions": quick_result.get("immediate_actions", [])},
                "escalation_recommendation": {},
                "processing_metadata": {
                    "analysis_type": "quick",
                    "analysis_completed_at": datetime.now().isoformat()
                },
            }
            
    except Exception as e:
        logger.error(f"AI compliance analysis failed: {e}")
        return {"error": str(e)}

@mcp.tool()
async def quick_ai_assessment(
    expense_id: str,
    amount: float,
    category: str,
    vendor: str,
    employee_name: str,
    department: str,
    description: str,
    receipt_attached: bool,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Quick AI assessment for expenses without full workflow.
    
    Args:
        expense_id: Unique identifier for the expense
        amount: Expense amount
        category: Expense category
        vendor: Vendor name
        employee_name: Name of the employee
        department: Employee department
        description: Expense description
        receipt_attached: Whether receipt is attached
        context: Additional context
        
    Returns:
        Quick assessment result
    """
    expense = {
        "expense_id": expense_id,
        "amount": amount,
        "category": category,
        "vendor": vendor,
        "employee_name": employee_name,
        "department": department,
        "description": description,
        "receipt_attached": receipt_attached
    }
    
    return await ai_compliance_service.quick_ai_assessment(expense, context)

@mcp.tool()
async def batch_analyze_compliance(
    expenses: List[Dict[str, Any]],
    full_analysis: bool = False
) -> Dict[str, Any]:
    """Batch process multiple expenses with AI analysis.
    
    Args:
        expenses: List of expense records to analyze
        full_analysis: Whether to run full analysis for each expense
        
    Returns:
        Batch analysis results
    """
    try:
        results = []
        for expense in expenses:
            if full_analysis:
                result = await ai_compliance_service.analyze_compliance_with_ai(expense)
            else:
                quick_result = await ai_compliance_service.quick_ai_assessment(expense)
                result = {
                    "expense_id": expense.get("expense_id"),
                    "quick_assessment": quick_result
                }
            results.append(result)
        
        return {
            "batch_id": f"batch_{uuid.uuid4().hex[:8]}",
            "processed_count": len(results),
            "results": results,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch AI analysis failed: {e}")
        return {"error": str(e)}

@mcp.tool()
async def get_employee_ai_insights(
    employee_id: str,
    days: int = 30
) -> Dict[str, Any]:
    """Get AI-generated insights for specific employee.
    
    Args:
        employee_id: Employee identifier
        days: Number of days to analyze
        
    Returns:
        Employee insights and recommendations
    """
    try:
        # Mock data for demonstration
        employee_data = {
            "employee_id": employee_id,
            "role": "Senior Analyst",
            "department": "Finance",
            "tenure_years": 3
        }
        
        expense_history = {
            "expenses": [
                {"amount": 45, "category": "meals", "date": "2024-01-15"},
                {"amount": 120, "category": "travel", "date": "2024-01-20"},
                {"amount": 35, "category": "supplies", "date": "2024-01-25"}
            ],
            "total_amount": 200,
            "expense_count": 3
        }
        
        spending_analytics = {
            "avg_expense": 66.67,
            "most_common_category": "meals",
            "trend": "stable"
        }
        
        # Generate AI insights
        if ai_compliance_service.llm_client:
            insights_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an AI analyst providing employee spending insights.
                Analyze the employee's expense patterns and provide actionable insights."""),
                
                HumanMessage(content=f"""
                Employee: {employee_id}
                Period: Last {days} days
                
                Employee Data: {json.dumps(employee_data, indent=2)}
                Expense History: {json.dumps(expense_history, indent=2)}
                Analytics: {json.dumps(spending_analytics, indent=2)}
                
                Provide insights on:
                - spending_trends
                - compliance_risk_factors
                - improvement_opportunities
                - personalized_recommendations
                """)
            ])
            
            response = await ai_compliance_service.llm_client.ainvoke(insights_prompt.format_messages())
            ai_insights = ai_compliance_service._parse_ai_response(response.content)
        else:
            ai_insights = {"error": "AI client not available"}
        
        return {
            "employee_id": employee_id,
            "analysis_period_days": days,
            "ai_insights": ai_insights,
            "data_sources": {
                "employee_data_available": bool(employee_data),
                "expense_history_count": len(expense_history.get("expenses", [])),
                "analytics_available": bool(spending_analytics)
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Employee AI insights failed: {e}")
        return {"error": str(e)}

@mcp.tool()
async def explain_ai_decision(
    expense_id: str,
    decision_type: str
) -> Dict[str, Any]:
    """Explain AI decision making process for transparency.
    
    Args:
        expense_id: Expense identifier
        decision_type: Type of decision to explain
        
    Returns:
        Detailed explanation of AI decision
    """
    try:
        if not ai_compliance_service.llm_client:
            return {"error": "AI explanation service not available"}
        
        explanation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI explainability assistant. Provide clear, 
            understandable explanations of AI compliance decisions for audit and transparency purposes."""),
            
            HumanMessage(content=f"""
            Explain the AI decision process for:
            - Expense ID: {expense_id}
            - Decision Type: {decision_type}
            
            Provide explanation covering:
            - reasoning_process: step-by-step logic
            - data_considered: what information was used
            - confidence_factors: what made the AI confident/uncertain
            - alternative_scenarios: what could change the decision
            - human_oversight_points: where human review is recommended
            """)
        ])
        
        response = await ai_compliance_service.llm_client.ainvoke(explanation_prompt.format_messages())
        explanation = ai_compliance_service._parse_ai_response(response.content)
        
        return {
            "expense_id": expense_id,
            "decision_type": decision_type,
            "explanation": explanation,
            "ai_model": AI_CONFIG["model"],
            "explained_at": datetime.now().isoformat(),
            "explainability_version": "1.0"
        }
        
    except Exception as e:
        logger.error(f"AI decision explanation failed: {e}")
        return {"error": str(e)}

# MCP Resources
@mcp.resource("workflow://status")
async def workflow_status() -> str:
    """Get status of AI compliance workflow components"""
    status = {
        "ai_client_available": ai_compliance_service.llm_client is not None,
        "langgraph_workflow_ready": ai_compliance_service.compliance_graph is not None,
        "ai_model": AI_CONFIG["model"],
        "workflow_version": "3.0.0",
        "capabilities": {
            "full_ai_analysis": ai_compliance_service.compliance_graph is not None,
            "quick_assessment": ai_compliance_service.llm_client is not None,
            "batch_processing": True,
            "employee_insights": True,
            "decision_explanation": ai_compliance_service.llm_client is not None,
        },
        "status_checked_at": datetime.now().isoformat()
    }
    return json.dumps(status, indent=2)

@mcp.resource("health://service")
async def health_status() -> str:
    """Enhanced health check with AI component status"""
    health = {
        "status": "healthy",
        "service": "ai-compliance",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "ai_client": {
                "status": "connected" if ai_compliance_service.llm_client else "disconnected",
                "model": AI_CONFIG["model"] if ai_compliance_service.llm_client else None
            },
            "langgraph_workflow": {
                "status": "ready" if ai_compliance_service.compliance_graph else "not_available",
                "version": "3.0.0"
            }
        },
        "capabilities": {
            "ai_powered_analysis": ai_compliance_service.llm_client is not None,
            "workflow_automation": ai_compliance_service.compliance_graph is not None,
            "explainable_ai": ai_compliance_service.llm_client is not None
        }
    }
    
    # Determine overall health
    if not ai_compliance_service.llm_client:
        health["status"] = "degraded"
        health["warnings"] = ["AI client not available"]
    
    return json.dumps(health, indent=2)

# MCP Prompts
@mcp.prompt()
def compliance_analysis_prompt(
    expense_id: str,
    amount: float,
    category: str
) -> str:
    """Generate a prompt for compliance analysis"""
    return f"""Please analyze expense {expense_id} for compliance:
    - Amount: ${amount}
    - Category: {category}
    
    Use the analyze_compliance_with_ai tool for comprehensive analysis."""

@mcp.prompt()
def employee_insights_prompt(employee_id: str) -> str:
    """Generate a prompt for employee insights"""
    return f"""Generate spending insights for employee {employee_id}.
    
    Use the get_employee_ai_insights tool to analyze their expense patterns."""

@mcp.prompt()
def explain_decision_prompt(expense_id: str) -> str:
    """Generate a prompt for decision explanation"""
    return f"""Explain the AI compliance decision for expense {expense_id}.
    
    Use the explain_ai_decision tool to provide transparency."""

# Lifecycle management
async def on_startup():
    """Initialize AI services on startup"""
    logger.info("AI Compliance Service starting up...")
    
    if ai_compliance_service.llm_client:
        logger.info("AI client ready")
    else:
        logger.warning("AI client not available - check API key")
    
    if ai_compliance_service.compliance_graph:
        logger.info("LangGraph compliance workflow ready")
    else:
        logger.warning("LangGraph workflow not initialized")
    
    logger.info("AI Compliance Service ready")

async def on_shutdown():
    """Cleanup on shutdown"""
    logger.info("AI Compliance Service shutting down...")
    logger.info("AI Compliance Service shutdown complete")

if __name__ == "__main__":
    # Run startup function before starting the server
    asyncio.run(on_startup())
    
    # Run the MCP server
    try:
        mcp.run()
    finally:
        # Run shutdown function when server stops
        asyncio.run(on_shutdown())

