# Use the official vLLM base image
FROM vllm/vllm-openai:latest

# Run the vLLM OpenAI-compatible server
ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server --model $MODEL_NAME $VLLM_ARGS
