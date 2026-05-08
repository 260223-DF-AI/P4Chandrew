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
  echo "Loading .env..."
  while IFS='=' read -r key value; do
    # Remove Windows carriage returns and whitespace
    clean_key=$(echo "$key" | tr -d '\r' | xargs)
    clean_value=$(echo "$value" | tr -d '\r' | xargs)
    
    # Skip empty lines or comments
    [[ -z "$clean_key" || "$clean_key" == \#* ]] && continue
    
    export "$clean_key"="$clean_value"
  done < .env
fi

echo "Building SAM application (Docker image)..."
sam build --template-file template.yaml

echo "Deploying to AWS..."
sam deploy \
  --stack-name researchflow \
  --capabilities CAPABILITY_IAM \
  --resolve-image-repos \
  --resolve-s3 \
  --parameter-overrides \
    PineconeApiKey="${PINECONE_API_KEY}" \
    PineconeIndexName="researchflow" \
    CohereApiKey="${COHERE_API_KEY}" \
    BedrockModelId="${BEDROCK_MODEL_ID}" \
    EmbeddingModelId="${BEDROCK_EMBEDDING_MODEL_ID}" \
    HitlConfidenceThreshold="${HITL_CONFIDENCE_THRESHOLD}" \
    MaxRefinementIterations="${MAX_REFINEMENT_ITERATIONS}"

echo "Deployment complete. Use the FunctionUrl from the output for your curl command."