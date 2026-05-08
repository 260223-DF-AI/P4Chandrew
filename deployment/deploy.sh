#!/usr/bin/env bash
# =============================================================================
# ResearchFlow — Deployment Script
# =============================================================================
# Builds and deploys the Lambda function using AWS SAM.
#
# Prerequisites:
#   - AWS CLI configured with valid credentials
#   - AWS SAM CLI installed (pip install aws-sam-cli)
#   - Docker installed (for sam build)
#
# Usage:
#   cd deployment && bash deploy.sh
# =============================================================================

set -euo pipefail

if [ -f .env ]; then
  echo "Loading variables from .env..."
  # Export variables while ignoring comments and empty lines
  export $(grep -v '^#' .env | xargs)
else
  echo "No .env file found. Proceeding with existing environment variables..."
fi

: "${PINECONE_API_KEY:?PINECONE_API_KEY env var must be set}"

echo "Building SAM application (Docker image)..."
sam build --template-file template.yaml

echo "Deploying to AWS..."
sam deploy \
  --guided \
  --stack-name researchflow \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    PineconeApiKey="${PINECONE_API_KEY}" \
    PineconeIndexName="${PINECONE_INDEX_NAME:-researchflow}" \
    BedrockModelId="${BEDROCK_MODEL_ID:-anthropic.claude-3-haiku-20240307-v1:0}" \
    EmbeddingModelId="${EMBEDDING_MODEL_ID:-amazon.titan-embed-text-v2:0}" \
    HitlConfidenceThreshold="${HITL_CONFIDENCE_THRESHOLD:-0.8}" \
    MaxRefinementIterations="${MAX_REFINEMENT_ITERATIONS:-3}"

echo "Deployment complete. Check the Outputs above for your API endpoint."