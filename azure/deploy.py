
import os
import time
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, command
from azure.ai.ml.entities import (
    Model,
    Environment,
    BuildContext,
    ManagedOnlineEndpoint,
    IdentityConfiguration,
    ManagedIdentityConfiguration,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings,
)
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

# Optional: OpenAI package for later usage
from openai import OpenAI

# Step 1: Authenticate & connect to workspace

credential = DefaultAzureCredential()
workspace_config_path = "/workspaces/DeepSeek-R1_distilled_azure_deployment/config.json" 
workspace_ml_client = MLClient.from_config(credential, config_path=workspace_config_path)

try:
    workspace_ml_client = MLClient.from_config(credential)
except Exception as ex:
    print(f"Error connecting to Azure ML Workspace: {ex}")
    exit(1)

subscription_id = workspace_ml_client.subscription_id
resource_group = workspace_ml_client.resource_group_name
workspace_name = workspace_ml_client.workspace_name

print("Connected to Azure ML Workspace:")
print(f"  Subscription ID: {subscription_id}")
print(f"  Resource Group:  {resource_group}")
print(f"  Workspace Name:  {workspace_name}")

# Step 2: Define model and endpoint names

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
endpoint_name = "deepseek-quen-15B-endpoint"

# Retrieve the model - assuming it has been registered already.
# If not, you could register it or update accordingly.
try:
    model = workspace_ml_client.models.get(model_name)
    print(f"Retrieved model: {model.name}")
except Exception as ex:
    print(f"Could not retrieve model '{model_name}'. Using model identifier instead.")
    model = model_name  # Fallback to string identifier if not registered

# Step 3: Create a custom Docker-based environment

env_docker_image = Environment(
    build=BuildContext(path="environment"),  
    name="vllm-custom",
    description="Environment created from a Docker context.",
    inference_config={
        "liveness_route": {"port": 8000, "path": "/health"},
        "readiness_route": {"port": 8000, "path": "/health"},
        "scoring_route": {"port": 8000, "path": "/"},
    },
)
env_asset = workspace_ml_client.environments.create_or_update(env_docker_image)
print(f"Environment '{env_asset.name}' registered successfully.")

# Step 4: Create or retrieve the Managed Online Endpoint

try:
    endpoint = workspace_ml_client.online_endpoints.get(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists. Using existing endpoint.")
except Exception:
    print(f"Endpoint '{endpoint_name}' not found. Creating a new endpoint.")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Test endpoint for deepseek",
    )
    workspace_ml_client.begin_create_or_update(endpoint).wait()
    print(f"Endpoint '{endpoint_name}' created successfully.")


# Step 5: Define environment variables for the deployment

env_vars = {
    "MODEL_NAME": model_name,
    "VLLM_ARGS": "--max-model-len 32768 --enforce-eager",
}
deployment_env_vars = {**env_vars}
print("Deployment environment variables configured:")
print(deployment_env_vars)


# Step 6: Create the Managed Online Deployment

t0 = time.time()
deployment = ManagedOnlineDeployment(
    name="deepseek-quen-15b-deployment",
    endpoint_name=endpoint_name,
    model=model,  # Either the registered Model object or its identifier
    instance_type="Standard_NC24ads_A100_v4",
    instance_count=1,
    environment_variables=deployment_env_vars,
    environment=env_asset,
    request_settings=OnlineRequestSettings(
        max_concurrent_requests_per_instance=2,
        request_timeout_ms=20000,
        max_queue_wait_ms=60000,
    ),
    liveness_probe=ProbeSettings(
        failure_threshold=5,
        success_threshold=1,
        timeout=10,
        period=30,
        initial_delay=120,
    ),
    readiness_probe=ProbeSettings(
        failure_threshold=30,
        success_threshold=1,
        timeout=2,
        period=10,
        initial_delay=120,
    ),
)

# Trigger deployment creation
try:
    workspace_ml_client.begin_create_or_update(deployment).wait()
    print("Deployment created successfully.")
except Exception as err:
    raise RuntimeError(
        f"Deployment creation failed. Detailed Response:\n{err}"
    ) from err

# Step 7: Update endpoint traffic to use this deployment

endpoint.traffic = {"deepseek-quen-15b-deployment": 100}
endpoint_poller = workspace_ml_client.online_endpoints.begin_create_or_update(endpoint)
endpoint_results = endpoint_poller.result()
print("Endpoint traffic updated successfully.")
print(f"Endpoint name: {endpoint_results.name}")


# Step 8: Retrieve endpoint keys and create an OpenAI-compatible client

keys = workspace_ml_client.online_endpoints.get_keys(name=endpoint_results.name)
primary_key = keys.primary_key

# Build the scoring URL from the endpoint result. The API path might vary.
scoring_url = os.path.join(endpoint_results.scoring_uri, "v1")

# Optionally, create an OpenAI client to interact with the endpoint.
client = OpenAI(base_url=scoring_url, api_key=primary_key)
print("OpenAI-compatible client created successfully.")

# Define the model path variable for further use if necessary.
model_path = model_name

t1 = time.time()
print(f"Total deployment process time: {t1 - t0:.2f} seconds")
