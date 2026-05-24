"""
Azure ML Managed Online Endpoint — Deployment Script

Usage:
    python deployment/azure/deploy.py

Prerequisites:
    pip install azure-ai-ml azure-identity

The script:
  1. Connects to your Azure ML workspace
  2. Registers the XGBoost model (models/ directory)
  3. Creates a managed online endpoint
  4. Deploys with 100 % traffic
  5. Smoke-tests the live endpoint
  6. Prints the scoring URI and API key
"""
import json
import time
import argparse
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

# ── Workspace config ────────────────────────────────────────────────────────
SUBSCRIPTION_ID = "8d75816b-cc88-4c0f-aa3c-2724c8649b94"
RESOURCE_GROUP  = "2445026-rg"
WORKSPACE_NAME  = "AL_ML_LEARNING_COURSE"

ENDPOINT_NAME   = "fraud-detection-endpoint"
DEPLOYMENT_NAME = "xgboost-blue"
MODEL_NAME      = "fraud-xgboost"
MODEL_VERSION   = "1"

REPO_ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR      = REPO_ROOT / "models"
SCORE_DIR       = Path(__file__).resolve().parent   # deployment/azure/
# ────────────────────────────────────────────────────────────────────────────


def get_ml_client(use_browser: bool = False) -> MLClient:
    credential = (
        InteractiveBrowserCredential() if use_browser else DefaultAzureCredential()
    )
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def register_model(client: MLClient) -> Model:
    print(f"\n[1/5] Registering model '{MODEL_NAME}' v{MODEL_VERSION} ...")
    model = Model(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        path=str(MODELS_DIR),
        description="XGBoost fraud detector — ROC-AUC 0.9518, trained on 232 984 transactions.",
        tags={"algorithm": "xgboost", "roc_auc": "0.9518", "project": "credit-fraud-detection"},
    )
    registered = client.models.create_or_update(model)
    print(f"    Model registered: {registered.name} v{registered.version}")
    return registered


def create_endpoint(client: MLClient) -> ManagedOnlineEndpoint:
    print(f"\n[2/5] Creating endpoint '{ENDPOINT_NAME}' ...")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Real-time fraud detection — XGBoost (ROC-AUC 0.9518)",
        auth_mode="key",
        tags={"project": "credit-fraud-detection", "model": "xgboost"},
    )
    poller = client.online_endpoints.begin_create_or_update(endpoint)
    result = poller.result()
    print(f"    Endpoint state: {result.provisioning_state}")
    return result


def create_deployment(client: MLClient, registered_model: Model) -> ManagedOnlineDeployment:
    print(f"\n[3/5] Creating deployment '{DEPLOYMENT_NAME}' ...")
    env = Environment(
        name="fraud-detection-env",
        conda_file=str(SCORE_DIR / "conda_env.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=registered_model.id,
        code_configuration=CodeConfiguration(
            code=str(SCORE_DIR),
            scoring_script="score.py",
        ),
        environment=env,
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )
    poller = client.online_deployments.begin_create_or_update(deployment)
    result = poller.result()
    print(f"    Deployment state: {result.provisioning_state}")
    return result


def set_traffic(client: MLClient) -> None:
    print(f"\n[4/5] Routing 100 % traffic to '{DEPLOYMENT_NAME}' ...")
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    print("    Traffic updated.")


def smoke_test(client: MLClient) -> None:
    print(f"\n[5/5] Running smoke test ...")
    sample = {
        "transactions": [
            {
                "TRANSACTION_ID": 99999,
                "TX_DATETIME":    "2026-05-24 21:30:00",
                "CUSTOMER_ID":    1,
                "TERMINAL_ID":    5,
                "TX_AMOUNT":      350.0,
                "x_customer_id":  25.0,
                "y_customer_id":  40.0,
                "mean_amount":    80.0,
                "std_amount":     30.0,
                "mean_nb_tx_per_day": 2.0,
                "nb_terminals":   4,
                "x_terminal_id":  50.0,
                "y_terminal_id":  60.0,
            }
        ]
    }
    response = client.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        request_file=None,
        request_body=json.dumps(sample),
    )
    print(f"    Response: {response}")


def print_endpoint_info(client: MLClient) -> None:
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    keys     = client.online_endpoints.get_keys(name=ENDPOINT_NAME)
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"  Scoring URI : {endpoint.scoring_uri}")
    print(f"  Primary key : {keys.primary_key}")
    print(f"  Swagger URI : {endpoint.openapi_uri}")
    print("=" * 60)
    print("\nSet these in your dashboard/.env:")
    print(f'  AZURE_ENDPOINT_URL="{endpoint.scoring_uri}"')
    print(f'  AZURE_ENDPOINT_KEY="{keys.primary_key}"')


def main(use_browser: bool = False) -> None:
    client           = get_ml_client(use_browser)
    registered_model = register_model(client)
    create_endpoint(client)
    create_deployment(client, registered_model)
    set_traffic(client)
    smoke_test(client)
    print_endpoint_info(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy fraud detection model to Azure ML")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Use interactive browser login instead of DefaultAzureCredential",
    )
    args = main(use_browser=parser.parse_args().browser)
