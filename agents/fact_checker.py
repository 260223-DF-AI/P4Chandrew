"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

import os
import re

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pydantic import BaseModel, Field

from agents.state import ResearchState


class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""
    claim: str
    verdict: str = Field(pattern=r"^(Supported|Unsupported|Inconclusive)$")
    evidence: str


class FactCheckReport(BaseModel):
    """Full verification report across all claims."""
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    status: str = Field(default="Pending", pattern=r"^(Pending|Accepted|Escalated)$")

_VERDICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict 2024 D&D Fact-Checker. Your goal is to verify claims against the provided evidence.\n\n"
        "CORE LOGIC - SPECIFIC BEATS GENERAL:\n"
        "1. If a claim describes a general 2024 rule (e.g., speed is 0), but the evidence describes "
        "a specific exception (e.g., a feat or monster trait that ignores that rule), "
        "you MUST mark the claim as 'Supported'. Note the exception in the evidence snippet.\n"
        "2. Only mark 'Unsupported' if the claim directly contradicts the core mechanics provided "
        "in the official evidence.\n\n"
        "VERDICT DEFINITIONS:\n"
        " • Supported: The evidence confirms the core mechanic of the claim.\n"
        " • Unsupported: The evidence explicitly contradicts the claim.\n"
        " • Inconclusive: The evidence does not mention the claim's topic at all.\n\n"
        "OUTPUT INSTRUCTIONS:\n"
        "Return a JSON object matching the requested schema. The 'verdict' MUST be exactly one of: "
        "Supported, Unsupported, or Inconclusive. The 'evidence' field must be a short, verbatim "
        "snippet from the provided text justifying your choice."
    )),
    ("human", "Claim: {claim}\n\nEvidence:\n{evidence}"),
])


class _SingleVerdict(BaseModel):
    """Schema the verdict-LLM is forced into."""
    verdict: str = Field(
        pattern=r"^(Supported|Unsupported|Inconclusive)$",
        description="Exactly one of: Supported, Unsupported, Inconclusive",
    )
    evidence: str = Field(
        description="A short quoted snippet from the evidence justifying the verdict",
    )


_embedder = None
_pinecone_index = None
_verdict_llm = None


def _lazy_init():
    """Module-level singletons; lets unit tests monkeypatch."""
    global _embedder, _pinecone_index, _verdict_llm
    if _embedder is None:
        _embedder = BedrockEmbeddings(
            model_id=os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            region_name=os.environ["AWS_REGION"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            model_kwargs={"dimensions": 1024},
        )
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    if _verdict_llm is None:
        _verdict_llm = ChatBedrock(
            model_id=os.environ.get(
                "BEDROCK_MODEL_ID",
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            ),
            region_name=os.environ["AWS_REGION"],
            model_kwargs={"max_tokens": 256, "temperature": 0.0},
        )


def _split_into_claims(answer: str) -> list[str]:
    """Heuristic claim extraction. Split on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s for s in sentences if len(s) > 20]


def _verify_claim(claim: str) -> ClaimVerdict:
    query_vec = _embedder.embed_query(claim)
    raw = _pinecone_index.query(
        vector=query_vec, top_k=5,
        namespace="fact-check-sources",
        include_metadata=True,
    )
    matches = raw.get("matches", []) if isinstance(raw, dict) else raw["matches"]
    
    if not matches:
        print(f"--- Fact-Checker: No matches in 'fact-check-sources' for claim: {claim[:50]}... ---")
        return ClaimVerdict(claim=claim, verdict="Inconclusive",
                            evidence="No supporting documents found.")

    evidence_block = "\n\n".join(
        m["metadata"].get("text", "") for m in matches
    )
    chain = _VERDICT_PROMPT | _verdict_llm.with_structured_output(_SingleVerdict)
    out: _SingleVerdict = chain.invoke({"claim": claim, "evidence": evidence_block})
    return ClaimVerdict(claim=claim, verdict=out.verdict, evidence=out.evidence)


def fact_checker_node(state: ResearchState) -> dict:
    """
    Verify the Analyst's response against trusted reference sources.

    TODO:
    - Extract claims from state["analysis"].
    - Query the 'fact-check-sources' Pinecone namespace for each claim.
    - Produce per-claim verdicts.
    - If confidence < threshold, trigger HITL interrupt.
    - Support Time Travel via state checkpointing.
    """
    _lazy_init()
    log = ["Fact Checker: starting verification"]

    analysis = state.get("analysis") or {}
    answer = analysis.get("answer", "")
    claims = _split_into_claims(answer)
    log.append(f"Fact Checker: extracted {len(claims)} claims")

    if not claims:
        report = FactCheckReport(verdicts=[], overall_confidence=0.0)
        return {
            "fact_check_report": report.model_dump(),
            "confidence_score": 0.0,
            "needs_hitl": True,
            "scratchpad": log + ["Fact Checker: no claims, escalating to HITL"],
        }

    verdicts = [_verify_claim(c) for c in claims]
    counts = {"Supported": 0, "Unsupported": 0, "Inconclusive": 0}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

    # Confidence = (supported - unsupported) / total, clamped to [0, 1].
    total = max(len(verdicts), 1)
    raw = (counts["Supported"] - counts["Unsupported"]) / total
    #overall = max(0.0, min(1.0, raw))
    overall = counts["Supported"] / total 
    threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    
    # Removed counts["Unsupported"] > 0 or
    if  overall < threshold:
        report_status = "Escalated"
        needs_hitl = True
    else:
        report_status = "Accepted"
        needs_hitl = False

    report = FactCheckReport(verdicts=verdicts, overall_confidence=overall, status=report_status)
    log.append(
        f"Fact Checker: supported={counts['Supported']}, "
        f"unsupported={counts['Unsupported']}, inconclusive={counts['Inconclusive']}, "
        f"overall={overall:.2f}, hitl={needs_hitl}"
    )
    
    for v in verdicts:
        if v.verdict == "Unsupported":
            print(f"\nFAILED CLAIM: {v.claim}")
            print(f"OFFICIAL EVIDENCE: {v.evidence}\n")

    return {
        "fact_check_report": report.model_dump(),
        "confidence_score": overall,
        "needs_hitl": needs_hitl,
        "scratchpad": log,
    }

    #Mock implementation
    # return {
    #     "fact_check_report": {
    #         "status": "Accepted", 
    #         "details": "Fact-check skipped for testing."
    #     },
    #     "scratchpad": state.get("scratchpad", []) + ["Fact-Checker: Bypassed. Not implemented."]
    # }
