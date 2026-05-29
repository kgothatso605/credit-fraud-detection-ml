"""
Deploy the fraud detection FastAPI app to Azure App Service.

Prerequisites:
    brew install azure-cli
    az login

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
APP_NAME       = "fraud-detect-api-2445026"
SKU            = "B1"
PYTHON_VERSION = "PYTHON|3.11"
REPO_ROOT      = Path(__file__).resolve().parents[2]
WEBAPP_SRC     = Path(__file__).resolve().parent / "webapp"
# ─────────────────────────────────────────────────────────────────────────────


def _brew_env() -> dict:
    """
    Fix the macOS/libexpat symbol mismatch in Azure CLI 2.86 on Python 3.13.
    Prepend Homebrew's lib directory so the CLI picks up the correct libexpat.
    """
    env = os.environ.copy()
    brew = subprocess.run(["brew", "--prefix"], capture_output=True, text=True)
    if brew.returncode == 0:
        lib = os.path.join(brew.stdout.strip(), "lib")
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else lib
    return env


_ENV = _brew_env()   # computed once at import time


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True, env=_ENV, **kwargs)
    if result.stdout.strip():
        # suppress large JSON blobs
        preview = result.stdout.strip()
        if len(preview) > 300:
            preview = preview[:300] + " ..."
        print(f"    {preview}")
    if result.returncode != 0:
        print(f"\n  ERROR (exit {result.returncode}):")
        print(f"  {result.stderr.strip()}")
        if check:
            sys.exit(1)
    return result


def check_az_cli() -> None:
    result = subprocess.run(
        ["az", "account", "show"], capture_output=True, text=True, env=_ENV
    )
    if result.returncode != 0:
        print("\nERROR: Not logged in to Azure CLI.  Run:  az login")
        sys.exit(1)
    account = json.loads(result.stdout)
    print(f"  Logged in as : {account.get('user', {}).get('name', '?')}")
    print(f"  Subscription : {account.get('name', '?')} ({account.get('id', '?')[:8]}...)")


def get_rg_location() -> str:
    r = run(
        ["az", "group", "show", "--name", RESOURCE_GROUP, "--query", "location", "-o", "tsv"]
    )
    loc = r.stdout.strip()
    print(f"    Resource group location: {loc}")
    return loc


def plan_existing_location() -> str | None:
    """Return location if the App Service Plan already exists, else None."""
    r = subprocess.run(
        [
            "az", "appservice", "plan", "show",
            "--name",           f"{APP_NAME}-plan",
            "--resource-group", RESOURCE_GROUP,
            "--query",          "location",
            "-o",               "tsv",
        ],
        capture_output=True, text=True, env=_ENV,
    )
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return None


def get_all_locations() -> list[str]:
    r = subprocess.run(
        ["az", "account", "list-locations", "--query", "[].name", "-o", "tsv"],
        text=True, capture_output=True, env=_ENV,
    )
    if r.returncode != 0:
        return []
    locs = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
    prefer = [
        "southafricanorth",   # known-allowed for this subscription
        "eastus", "westus", "westus2", "centralus",
        "northcentralus", "southcentralus", "westus3",
        "northeurope", "westeurope", "uksouth", "ukwest",
        "canadacentral", "canadaeast",
        "australiaeast", "australiasoutheast",
        "southeastasia", "eastasia", "japaneast", "japanwest",
        "brazilsouth", "eastus2",
    ]
    ordered = [l for l in prefer if l in locs]
    ordered += [l for l in locs if l not in ordered]
    return ordered


def find_or_create_plan() -> str:
    """Return the region of the App Service Plan, creating it if needed."""
    existing = plan_existing_location()
    if existing:
        print(f"    App Service Plan already exists in '{existing}' — skipping create.")
        return existing

    all_locs = get_all_locations() or [
        "southafricanorth", "eastus", "westus", "northeurope", "westeurope"
    ]
    plan_name = f"{APP_NAME}-plan"
    print(f"  Probing {len(all_locs)} regions ...")

    for loc in all_locs:
        probe = subprocess.run(
            [
                "az", "appservice", "plan", "create",
                "--name",           plan_name,
                "--resource-group", RESOURCE_GROUP,
                "--location",       loc,
                "--sku",            SKU,
                "--is-linux",
            ],
            text=True, capture_output=True, env=_ENV,
        )
        if probe.returncode == 0:
            print(f"    Allowed region: {loc}")
            return loc

        err = probe.stderr.lower()
        if any(k in err for k in ("disallowed", "policy", "not available", "not supported")):
            print(f"    {loc}: blocked by policy")
            continue
        if "already exists" in err:
            print(f"    Plan already exists in {loc}")
            return loc
        print(f"\n  Unexpected error ({loc}):\n  {probe.stderr.strip()}")
        sys.exit(1)

    print("\nERROR: No allowed region found. Ask your Azure administrator.")
    print("Alternatively, run the dashboard with local inference:")
    print("  streamlit run dashboard/app.py  (click Start — no endpoint needed)")
    sys.exit(1)


def build_deploy_package() -> Path:
    tmp = Path(tempfile.mkdtemp()) / "fraud_api"
    tmp.mkdir(parents=True)

    for f in WEBAPP_SRC.iterdir():
        shutil.copy2(f, tmp / f.name)

    shutil.copytree(REPO_ROOT / "models", tmp / "models")
    shutil.copytree(REPO_ROOT / "src",    tmp / "src")

    (tmp / "startup.sh").write_text(
        "#!/bin/bash\n"
        "pip install -r /home/site/wwwroot/requirements.txt --quiet\n"
        "MODELS_DIR=/home/site/wwwroot/models "
        "uvicorn app:app --host 0.0.0.0 --port 8000\n"
    )
    os.chmod(tmp / "startup.sh", 0o755)

    zip_path = Path(tempfile.mkdtemp()) / "fraud_api.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(tmp))
    print(f"  Package: {zip_path}  ({zip_path.stat().st_size // 1024 // 1024} MB)")
    return zip_path


_PYEXPAT_CRASH = "pyexpat"   # signature of the known Azure CLI 2.86 macOS bug


def _is_pyexpat_crash(stderr: str) -> bool:
    """Azure CLI 2.86 crashes fetching FTP publish profiles on macOS/Python 3.13.
    The resource is created successfully before the crash — treat as non-fatal."""
    return _PYEXPAT_CRASH in stderr or "xmltodict" in stderr or "XML_SetAllocTracker" in stderr


def webapp_exists() -> bool:
    r = subprocess.run(
        ["az", "webapp", "show", "--name", APP_NAME,
         "--resource-group", RESOURCE_GROUP, "-o", "none"],
        capture_output=True, text=True, env=_ENV,
    )
    # returncode 0 → exists | pyexpat crash → CLI bug but resource may exist
    if r.returncode == 0:
        return True
    if _is_pyexpat_crash(r.stderr):
        # Try a simpler existence check via list
        r2 = subprocess.run(
            ["az", "webapp", "list", "--resource-group", RESOURCE_GROUP,
             "--query", f"[?name=='{APP_NAME}'].name", "-o", "tsv"],
            capture_output=True, text=True, env=_ENV,
        )
        return APP_NAME in r2.stdout
    return False


def deploy(zip_path: Path) -> str:
    plan_name = f"{APP_NAME}-plan"

    if webapp_exists():
        print(f"\n[2/4] Web App '{APP_NAME}' already exists — skipping create.")
    else:
        print(f"\n[2/4] Creating Web App '{APP_NAME}' ...")
        r = subprocess.run(
            [
                "az", "webapp", "create",
                "--name",           APP_NAME,
                "--resource-group", RESOURCE_GROUP,
                "--plan",           plan_name,
                "--runtime",        PYTHON_VERSION,
            ],
            capture_output=True, text=True, env=_ENV,
        )
        if r.returncode == 0:
            print("    Web App created.")
        elif _is_pyexpat_crash(r.stderr) or "already exists" in r.stderr:
            # CLI crashes fetching FTP URL after successful creation — harmless
            print("    Web App created (CLI crashed on non-critical FTP URL step — OK).")
        else:
            print(f"\n  ERROR:\n  {r.stderr.strip()}")
            sys.exit(1)

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
    print("\nPaste into the dashboard sidebar (leave API Key blank):")
    print(f"  {url}/score")


def main() -> None:
    print("\n=== Fraud Detection API — Azure App Service Deployment ===\n")
    check_az_cli()
    get_rg_location()

    print("\n[1/4] Ensuring App Service Plan exists ...")
    find_or_create_plan()

    zip_path = build_deploy_package()
    try:
        url = deploy(zip_path)
        print_result(url)
    finally:
        shutil.rmtree(zip_path.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
