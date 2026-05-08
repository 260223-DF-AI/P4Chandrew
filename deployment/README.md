## Code to run:
sam build
./deploy.sh

## If deployment fails
aws cloudformation delete-stack --stack-name researchflow
aws cloudformation describe-stacks --stack-name researchflow (If throws an error, it was successfully deleted)

## Logs
sam logs -n ResearchFunction --stack-name researchflow --tail

## curl

curl -X POST https://i9zzg9xtv9.execute-api.us-east-1.amazonaws.com/Prod/research \
  -H "Content-Type: application/json" \
  -d '{"question": "How many classes are there?", "user_id": "local_dev"}'

curl -X POST https://7be5xnhsdksueqmx3pxznwvjdq0wqdnt.lambda-url.us-east-1.on.aws/ \
  -H "Content-Type: application/json" \
  -d '{"question": "How many classes are there?", "user_id": "local_dev"}'