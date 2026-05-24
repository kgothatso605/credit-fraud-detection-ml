"""
Deploy the fraud detection FastAPI app to Azure App Service.

This approach avoids the Azure ML ACR dependency entirely.

Prerequisites:
    brew install azure-cli
    az login

Usage:
    python deployment/azure/deploy_webapp.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
RESOURCE_GROUP  = "2445026-rg"
APP_NAME        = "fraud-detection-api"       # must be globally unique
LOCATION        = "southafricanorth"           # closest Azure region to Wits
SKU             = "B1"                         # Basic tier — free for students
PYTHON_VERSION  = "PYTHON|3.11"
REPO_ROOT       = Path(__file__).resolve().parents[2]
WEBAPP_SRC      = Path(__file__).resolve().parent / "webapp"
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    return result


def check_az_cli() -> None:
    try:
        run(["az", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\nERROR: Azure CLI not found.")
        print("Install it with:  brew install azure-cli")
        print("Then log in:      az login")
        sys.exit(1)


def build_deploy_package() -> Path:
    """
    Copy webapp source + models into a temp directory.
    az webapp up will zip and upload this directory.
    """
    tmp = Path(tempfile.mkdtemp()) / "fraud_api"
    tmp.mkdir(parents=True)

    # Copy webapp source files
    for f in WEBAPP_SRC.iterdir():
        shutil.copy2(f, tmp / f.name)

    # Copy models (needed at runtime)
    models_dst = tmp / "models"
    shutil.copytree(REPO_ROOT / "models", models_dst)

    # Copy src/ (feature engineering)
    src_dst = tmp / "src"
    shutil.copytree(REPO_ROOT / "src", src_dst)

    # Write startup command file
    (tmp / "startup.sh").write_text(
        "#!/bin/bash\n"
        "pip install -r requirements.txt --quiet\n"
        f"MODELS_DIR=/home/site/wwwroot/models uvicorn app:app --host 0.0.0.0 --port 8000\n"
    )
    os.chmod(tmp / "startup.sh", 0o755)

    print(f"  Deploy package prepared at: {tmp}")
    return tmp


def deploy(package_dir: Path) -> str:
    """Create App Service Plan + Web App and deploy code."""

    print("\n[1/4] Creating App Service Plan ...")
    run([
        "az", "appservice", "plan", "create",
        "--name",           f"{APP_NAME}-plan",
        "--resource-group", RESOURCE_GROUP,
        "--location",       LOCATION,
        "--sku",            SKU,
        "--is-linux",
    ])

    print("\n[2/4] Creating Web App ...")
    run([
        "az", "webapp", "create",
        "--name",           APP_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--plan",           f"{APP_NAME}-plan",
        "--runtime",        PYTHON_VERSION,
    ])

    print("\n[3/4] Configuring startup & environment ...")
    run([
        "az", "webapp", "config", "set",
        "--name",            APP_NAME,
        "--resource-group",  RESOURCE_GROUP,
        "--startup-file",    "startup.sh",
    ])
    run([
        "az", "webapp", "config", "appsettings", "set",
        "--name",           APP_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--settings",
        "MODELS_DIR=/home/site/wwwroot/models",
        "SCM_DO_BUILD_DURING_DEPLOYMENT=true",
    ])

    print("\n[4/4] Deploying code (this may take 2–3 min) ...")
    run([
        "az", "webapp", "deploy",
        "--name",           APP_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--src-path",       str(package_dir),
        "--type",           "zip",
    ], cwd=str(package_dir))

    url = f"https://{APP_NAME}.azurewebsites.net"
    return url


def print_result(url: str) -> None:
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"  App URL    : {url}")
    print(f"  Score URL  : {url}/score")
    print(f"  Health URL : {url}/")
    print(f"  API docs   : {url}/docs")
    print("=" * 60)
    print("\nSet this in dashboard/.env:")
    print(f'  AZURE_ENDPOINT_URL="{url}/score"')
    print("  AZURE_ENDPOINT_KEY=  (leave blank — App Service uses no API key)")
    print("\nUpdate dashboard/utils/azure_client.py to use Bearer-free auth,")
    print("or simply leave the key blank and the client falls back to local model.")


def main() -> None:
    check_az_cli()
    print("\n=== Fraud Detection API — Azure App Service Deployment ===\n")

    pkg = build_deploy_package()
    try:
        url = deploy(pkg)
        print_result(url)
    finally:
        shutil.rmtree(pkg.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
