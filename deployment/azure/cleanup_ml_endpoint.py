"""
Delete the failed Azure ML managed online endpoint and its deployment.
Run this once to clean up the workspace after the ACR error.

Usage:
    python deployment/azure/cleanup_ml_endpoint.py --browser
"""
import argparse
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential, DefaultAzureCredential

SUBSCRIPTION_ID = "8d75816b-cc88-4c0f-aa3c-2724c8649b94"
RESOURCE_GROUP  = "2445026-rg"
WORKSPACE_NAME  = "AL_ML_LEARNING_COURSE"
ENDPOINT_NAME   = "fraud-detection-endpoint"


def main(use_browser: bool = False) -> None:
    credential = (
        InteractiveBrowserCredential() if use_browser else DefaultAzureCredential()
    )
    client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    print(f"Deleting endpoint '{ENDPOINT_NAME}' (and all deployments under it) ...")
    client.online_endpoints.begin_delete(name=ENDPOINT_NAME).result()
    print("Done — endpoint deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", action="store_true")
    main(use_browser=parser.parse_args().browser)
