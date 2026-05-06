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
        raw_body = event.get("body") or "{}"

        if isinstance(raw_body, str):
            body = json.loads(raw_body)
        else:
            body = raw_body

        question = (body.get("question") or "").strip()
        if not question:
            return _response(400, {"error": "Missing 'question' in request body."})
        
    except Exception as e:
        logger.exception("Research request failed.")
        return _response(500, {"error": str(e)})
