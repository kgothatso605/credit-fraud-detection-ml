"""
Deploy the fraud detection FastAPI app to Azure App Service.

This approach avoids the Azure ML ACR dependency entirely.

Prerequisites:
    brew install azure-cli
    az login                  ← must run this FIRST in your terminal

Usage:
    python deployment/azure/deploy_webapp.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
RESOURCE_GROUP = "2445026-rg"
APP_NAME       = "fraud-detect-api-2445026"   # globally unique; change if taken
SKU            = "B1"
PYTHON_VERSION = "PYTHON|3.11"
REPO_ROOT      = Path(__file__).resolve().parents[2]
WEBAPP_SRC     = Path(__file__).resolve().parent / "webapp"
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run az command and always show stderr on failure."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"\n  ERROR (exit {result.returncode}):")
        print(f"  {result.stderr.strip()}")
        if check:
            sys.exit(1)
    return result


def get_rg_location() -> str:
    """Read location from the existing resource group."""
    result = run(
        ["az", "group", "show", "--name", RESOURCE_GROUP, "--query", "location", "-o", "tsv"],
        check=True,
    )
    location = result.stdout.strip()
    print(f"    Resource group location: {location}")
    return location


def check_az_cli() -> None:
    result = subprocess.run(["az", "account", "show"], capture_output=True, text=True)
    if result.returncode != 0:
        print("\nERROR: Not logged in to Azure CLI.")
        print("Run:  az login")
        print("Then re-run this script.")
        sys.exit(1)
    account = json.loads(result.stdout)
    print(f"  Logged in as: {account.get('user', {}).get('name', '?')}")
    print(f"  Subscription: {account.get('name', '?')} ({account.get('id', '?')[:8]}...)")


def build_deploy_package() -> Path:
    """Bundle webapp source + models + src into a temp directory."""
    tmp = Path(tempfile.mkdtemp()) / "fraud_api"
    tmp.mkdir(parents=True)

    for f in WEBAPP_SRC.iterdir():
        shutil.copy2(f, tmp / f.name)

    shutil.copytree(REPO_ROOT / "models", tmp / "models")
    shutil.copytree(REPO_ROOT / "src",    tmp / "src")

    # Azure App Service startup command
    (tmp / "startup.sh").write_text(
        "#!/bin/bash\n"
        "pip install -r /home/site/wwwroot/requirements.txt --quiet\n"
        "MODELS_DIR=/home/site/wwwroot/models "
        "uvicorn app:app --host 0.0.0.0 --port 8000\n"
    )
    os.chmod(tmp / "startup.sh", 0o755)

    # Zip the package
    zip_path = Path(tempfile.mkdtemp()) / "fraud_api.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(tmp))

    print(f"  Package: {zip_path}  ({zip_path.stat().st_size // 1024 // 1024} MB)")
    return zip_path


def deploy(zip_path: Path, location: str) -> str:
    plan_name = f"{APP_NAME}-plan"

    print(f"\n[1/4] Creating App Service Plan '{plan_name}' in {location} ...")
    run([
        "az", "appservice", "plan", "create",
        "--name",           plan_name,
        "--resource-group", RESOURCE_GROUP,
        "--location",       location,
        "--sku",            SKU,
        "--is-linux",
    ])

    print(f"\n[2/4] Creating Web App '{APP_NAME}' ...")
    run([
        "az", "webapp", "create",
        "--name",           APP_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--plan",           plan_name,
        "--runtime",        PYTHON_VERSION,
    ])

    print("\n[3/4] Configuring startup command & env vars ...")
    run([
        "az", "webapp", "config", "set",
        "--name",           APP_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--startup-file",   "startup.sh",
    ])
    run([
        "az", "webapp", "config", "appsettings", "set",
        "--name",           APP_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--settings",
        "MODELS_DIR=/home/site/wwwroot/models",
        "SCM_DO_BUILD_DURING_DEPLOYMENT=false",
    ])

    print("\n[4/4] Uploading code (may take 2–4 min) ...")
    run([
        "az", "webapp", "deploy",
        "--name",           APP_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--src-path",       str(zip_path),
        "--type",           "zip",
        "--async",          "false",
    ])

    return f"https://{APP_NAME}.azurewebsites.net"


def print_result(url: str) -> None:
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"  Health check : {url}/")
    print(f"  Score URL    : {url}/score")
    print(f"  API docs     : {url}/docs")
    print("=" * 60)
    print("\nPaste into dashboard sidebar (API Key: leave blank):")
    print(f"  {url}/score")


def main() -> None:
    print("\n=== Fraud Detection API — Azure App Service Deployment ===\n")

    check_az_cli()
    location = get_rg_location()
    zip_path = build_deploy_package()

    try:
        url = deploy(zip_path, location)
        print_result(url)
    finally:
        shutil.rmtree(zip_path.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
