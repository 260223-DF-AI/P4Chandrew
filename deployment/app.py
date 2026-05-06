"""
ResearchFlow — AWS Lambda Handler

Entry point for the serverless deployment behind API Gateway.
Receives a POST /research request and invokes the Supervisor graph.
"""

import json
import logging
import os
import uuid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_graph = None


def _get_graph():
    """Initialize and return the Supervisor graph instance."""
    global _graph
    if _graph is None:
        # importing when needed to reduce cold start cost.
        # only paid once per container, not every init.
        from agents.supervisor import build_supervisor_graph
        _graph = build_supervisor_graph()
    
    return _graph


def _response(status_code: int, body: dict) -> dict:
    """Formats Lambda proxy integration response."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda handler for the /research endpoint.

    TODO:
    - Parse the request body from event.
    - Extract the "question" field.
    - Initialize and invoke the Supervisor graph.
    - Return the structured research report as JSON.
    - Handle errors gracefully with appropriate HTTP status codes.

    Expected request:  { "question": "..." }
    Expected response: { "statusCode": 200, "body": "<JSON report>" }
    """
    try:
        # parse and validate input
        raw_body = event.get("body") or "{}"

        if isinstance(raw_body, str):
            body = json.loads(raw_body)
        else:
            body = raw_body

        question = (body.get("question") or "").strip()
        if not question:
            return _response(400, {"error": "Missing 'question' in request body."})
        
        from middleware.pii_masking import mask_pii
        from middleware.guardrails import sanitize_input, detect_injection

        # check for possible injection patterns before sanitization
        if detect_injection(question):
            return _response(400, {"error": "Potential prompt injection detected in input."})
        question = mask_pii(sanitize_input(question))

        # invoke the Supervisor
        graph = _get_graph()
        config = {
            "configurable": {
                "thread_id": f"lambda-{uuid.uuid4()}"
            }
        }
        result = graph.invoke({
            "question": question,
            "user_id": body.get("user_id", "anonymous")
        }, config=config)

        analysis = result.get("analysis", {}) or {}
        return _response(200, {
            "answer": mask_pii(analysis.get("answer", "")),
            "citations": analysis.get("citations", []),
            "confidence": result.get("confidence_score", 0.0),
            "fact_check": result.get("fact_check_report", {}),
            "iterations": result.get("iteration_count", 0)
        })
        
    except Exception as e:
        logger.exception("Research request failed.")
        return _response(500, {"error": str(e)})
