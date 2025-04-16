import os
from openai import OpenAI

# -------------------------------
# Configure your endpoint details
# -------------------------------
# For testing, you can pass the values using environment variables.
# Ensure you have defined these environment variables or update these variables directly.
api_key = os.environ.get("OPENAI_API_KEY")         # Set this to your endpoint's primary key
base_url = os.environ.get("SCORING_URL")             # Set this to your endpoint scoring URI, e.g., "https://<your-endpoint>/v1"
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Must match the model path used in deployment

# -------------------------------
# Create your prompt messages
# -------------------------------
system_message = "You are a helpful Assistant"
user_message = "write a python code to read csv"

# -------------------------------
# Create an OpenAI-compatible client
# -------------------------------
# The OpenAI client here is configured to use your custom endpoint.
client = OpenAI(base_url=base_url, api_key=api_key)

# -------------------------------
# Create a chat completion request with streaming enabled
# -------------------------------
response = client.chat.completions.create(
    model=model_path,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ],
    temperature=0.7,
    max_tokens=4000,
    stream=True,  # Stream the response back chunk by chunk
)

print("Streaming response:")
for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, "content"):
        print(delta.content, end="", flush=True)
