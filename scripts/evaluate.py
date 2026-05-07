"""
ResearchFlow — RAGAS Evaluation Pipeline

Loads a golden dataset and runs a formal RAGAS evaluation measuring
faithfulness, answer relevancy, and context precision.

Usage:
    python scripts/evaluate.py --golden-dataset ./data/golden_dataset.json
    python -m scripts.evaluate --golden-dataset ./data/golden_dataset.json
"""

from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from agents.supervisor import build_supervisor_graph
from dotenv import load_dotenv
import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation.")
    parser.add_argument(
        "--golden-dataset",
        type=str,
        required=True,
        help="Path to the golden dataset JSON file.",
    )
    return parser.parse_args()


def load_golden_dataset(filepath: str) -> list[dict]:
    """
    Load the golden dataset from a JSON file.

    Expected format: see data/golden_dataset.json for the schema.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_predictions(dataset: list[dict]) -> list[dict]:
    """
    Run each question through the ResearchFlow pipeline and collect predictions.

    TODO:
    - For each entry in the dataset, invoke the Supervisor graph.
    - Capture the generated answer and the retrieved contexts.
    - Return a list of dicts with keys: question, answer, contexts.
    """
    graph = build_supervisor_graph()
    out = []
    for i, entry in enumerate(dataset):
        config = {"configurable": {"thread_id": f"eval-{i}"}}
        try:
            result = graph.invoke(
                {"question": entry["question"], "user_id": "evaluator"},
                config=config,
            )
        except Exception as e:
            print(f"  [warn] entry {i} failed: {e}")
            out.append({"question": entry["question"], "answer": "", "contexts": []})
            continue
        analysis = result.get("analysis", {}) or {}
        contexts = [c["content"] for c in result.get("retrieved_chunks", [])]
        out.append({
            "question": entry["question"],
            "answer": analysis.get("answer", ""),
            "contexts": contexts,
            "ground_truth": entry["ground_truth_answer"],
        })
        print(f"  [{i+1}/{len(dataset)}] done")
    return out


def run_ragas_evaluation(predictions: list[dict], golden: list[dict]) -> dict:
    """
    Evaluate predictions against the golden dataset using RAGAS.

    TODO:
    - Construct a RAGAS Dataset from predictions and ground truth.
    - Evaluate with metrics: faithfulness, answer_relevancy, context_precision.
    - Return a dict of metric_name → score.
    """
    
    bedrock_chat = ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        region_name=os.getenv("AWS_REGION", "us-east-1") 
    )
    bedrock_embeddings = BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_kwargs={"max_tokens": 9000, "temperature": 0.0}
    )
    # Reduce max_workers since it is throttling on the default
    run_config = RunConfig(max_workers=2, timeout=60) 
    # Wrap LLM models for Ragas
    ragas_llm = LangchainLLMWrapper(bedrock_chat)
    ragas_embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)
    
    ds = Dataset.from_list(predictions)

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )
    
    #TODO: result.scores.items() not working, as well as results._scores_dict.items()
    return {k: float(v) for k, v in result.scores.items()}


def main() -> None:
    """Orchestrate the evaluation pipeline."""
    load_dotenv()
    args = parse_args()

    golden = load_golden_dataset(args.golden_dataset)
    predictions = generate_predictions(golden)
    results = run_ragas_evaluation(predictions, golden)

    print("\n📊 RAGAS Evaluation Results:")
    print("-" * 40)
    for metric, score in results.items():
        print(f"  {metric:<25} {score:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
