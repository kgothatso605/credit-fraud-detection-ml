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


def get_all_locations() -> list[str]:
    """Return every Azure region available to this subscription."""
    r = subprocess.run(
        ["az", "account", "list-locations", "--query", "[].name", "-o", "tsv"],
        text=True, capture_output=True,
    )
    if r.returncode != 0:
        return []
    locs = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
    # Prefer cheaper / well-known regions first
    prefer = [
        "eastus", "westus", "westus2", "centralus", "northcentralus",
        "southcentralus", "westus3", "northeurope", "westeurope",
        "uksouth", "ukwest", "canadacentral", "canadaeast",
        "australiaeast", "australiasoutheast", "southeastasia",
        "eastasia", "japaneast", "japanwest", "brazilsouth",
        "eastus2",
    ]
    ordered = [l for l in prefer if l in locs]
    ordered += [l for l in locs if l not in ordered]
    return ordered


def find_allowed_location() -> str:
    """
    Azure for Students restricts deployments via policy.
    Probe every available region until one accepts the App Service Plan.
    Also creates a new resource group in that region if needed.
    """
    all_locs = get_all_locations()
    if not all_locs:
        all_locs = ["eastus", "westus", "westus2", "northeurope", "westeurope"]

    plan_name  = f"{APP_NAME}-plan"
    # We may need a separate RG in the allowed region
    deploy_rg  = RESOURCE_GROUP

    print(f"  Probing {len(all_locs)} regions (this may take ~1 min) ...")
    for loc in all_locs:
        probe = subprocess.run(
            [
                "az", "appservice", "plan", "create",
                "--name",           plan_name,
                "--resource-group", deploy_rg,
                "--location",       loc,
                "--sku",            SKU,
                "--is-linux",
            ],
            text=True, capture_output=True,
        )
        if probe.returncode == 0:
            print(f"    Allowed region: {loc}  (resource group: {deploy_rg})")
            return loc

        err = probe.stderr.lower()
        if "disallowed" in err or "policy" in err or "not available" in err:
            print(f"    {loc}: blocked")
            continue
        if "already exists" in err:
            # Plan already created in a previous run
            print(f"    Plan already exists in {loc}")
            return loc
        # Unexpected error — show it
        print(f"\n  Unexpected error in {loc}:\n  {probe.stderr.strip()}")
        sys.exit(1)

    print("\nERROR: No allowed region found after checking all available locations.")
    print("\nYour Azure for Students subscription has very restrictive policies.")
    print("Options:")
    print("  1. Ask your course administrator which region is allowed.")
    print("  2. Run the dashboard in local-model mode (no endpoint needed):")
    print("       streamlit run dashboard/app.py")
    print("     Then click Start — it uses the local XGBoost model automatically.")
    sys.exit(1)


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


def deploy(zip_path: Path) -> str:
    plan_name = f"{APP_NAME}-plan"
    # App Service Plan is already created by find_allowed_location()

    print(f"\n[2/4] Creating Web App '{APP_NAME}' ...")  # plan already exists
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
    get_rg_location()   # informational only

    print("\n[1/4] Finding an allowed region and creating App Service Plan ...")
    find_allowed_location()   # creates the plan in the first allowed region

    zip_path = build_deploy_package()
    try:
        url = deploy(zip_path)
        print_result(url)
    finally:
        shutil.rmtree(zip_path.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
