#!/usr/bin/env python3
"""
Development CLI for Databricks Asset Bundle workflow.

Usage:
    python scripts/dev.py <command> [options]

Commands:
    setup     - Check prerequisites
    unit      - Run local pytest
    validate  - Validate DAB config
    deploy    - Deploy bundle to workspace
    run       - Run the job (full mode)
    smoke     - Run the job (smoke mode - fast/cheap)
    destroy   - Tear down deployed resources

Authentication:
    Uses Databricks unified auth. Set profile via:
    - --profile <NAME>
    - DATABRICKS_CONFIG_PROFILE env var
    - Default profile in ~/.databrickscfg
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Output markers for machine parsing
MARKER_RUN_ID = "RUN_ID"
MARKER_RUN_URL = "RUN_URL"
MARKER_RESULT_JSON = "RESULT_JSON"
MARKER_FINAL_STATE = "FINAL_STATE"


def print_marker(key: str, value: str) -> None:
    """Print machine-readable marker line."""
    print(f"{key}={value}")


def run_cmd(
    cmd: list[str],
    capture: bool = False,
    check: bool = True,
    env_extra: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a shell command with optional environment additions."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    
    print(f">>> {' '.join(cmd)}")
    
    if capture:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
    else:
        result = subprocess.run(cmd, env=env)
    
    if check and result.returncode != 0:
        if capture and result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    
    return result


def get_profile_env(profile: str | None) -> dict:
    """Get environment dict for Databricks profile."""
    if profile:
        return {"DATABRICKS_CONFIG_PROFILE": profile}
    return {}


def cmd_setup(args: argparse.Namespace) -> None:
    """Check prerequisites are installed."""
    print("Checking prerequisites...\n")
    
    # Check Databricks CLI
    if shutil.which("databricks"):
        result = run_cmd(["databricks", "--version"], capture=True)
        print(f"✓ Databricks CLI: {result.stdout.strip()}")
    else:
        print("✗ Databricks CLI not found")
        print("  Install: https://docs.databricks.com/dev-tools/cli/install.html")
        sys.exit(1)
    
    # Check auth
    env = get_profile_env(args.profile)
    result = run_cmd(
        ["databricks", "auth", "describe"],
        capture=True,
        check=False,
        env_extra=env,
    )
    if result.returncode == 0:
        print(f"✓ Auth configured")
        # Extract host from output
        for line in result.stdout.splitlines():
            if "Host:" in line:
                print(f"  {line.strip()}")
    else:
        print("✗ Auth not configured")
        print("  Run: databricks auth login --host <WORKSPACE_URL>")
        sys.exit(1)
    
    # Check pytest
    if shutil.which("pytest"):
        print("✓ pytest available")
    else:
        print("⚠ pytest not found (install with: pip install pytest)")
    
    print("\n✓ Setup complete!")


def cmd_unit(args: argparse.Namespace) -> None:
    """Run local unit tests."""
    pytest_args = ["pytest", "tests/unit", "-v"]
    if args.coverage:
        pytest_args.extend(["--cov=src", "--cov-report=term-missing"])
    run_cmd(pytest_args)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate DAB configuration."""
    env = get_profile_env(args.profile)
    run_cmd(
        ["databricks", "bundle", "validate", "-t", args.target],
        env_extra=env,
    )
    print("\n✓ Bundle configuration valid!")


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy bundle to workspace."""
    env = get_profile_env(args.profile)
    run_cmd(
        ["databricks", "bundle", "deploy", "-t", args.target],
        env_extra=env,
    )
    print("\n✓ Bundle deployed!")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the job and wait for completion."""
    env = get_profile_env(args.profile)
    
    # Build job parameters
    job_params = {
        "env": args.env or args.target,
        "run_mode": args.run_mode,
        "json_params": args.json_params,
    }
    
    # Start the run
    cmd = [
        "databricks", "bundle", "run",
        "-t", args.target,
        "generate_data_job",
        "--params", json.dumps(job_params),
    ]
    
    print(f"\nStarting job with params: {json.dumps(job_params, indent=2)}\n")
    
    result = run_cmd(cmd, capture=True, check=False, env_extra=env)
    stdout = result.stdout
    stderr = result.stderr
    
    if result.returncode != 0:
        print(f"Job submission failed:\n{stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    
    # Parse run info from output
    # databricks bundle run outputs JSON or text depending on version
    run_id = None
    run_url = None
    
    # Try to extract run_id and URL from output
    # Format varies - try common patterns
    for line in stdout.splitlines():
        if "run_id" in line.lower() or "runid" in line.lower():
            match = re.search(r'(\d{10,})', line)
            if match:
                run_id = match.group(1)
        if "http" in line and "/jobs/" in line:
            match = re.search(r'(https://[^\s]+)', line)
            if match:
                run_url = match.group(1)
    
    # Also try JSON parsing
    try:
        data = json.loads(stdout)
        run_id = run_id or str(data.get("run_id", ""))
        run_url = run_url or data.get("run_page_url", "")
    except json.JSONDecodeError:
        pass
    
    if run_id:
        print_marker(MARKER_RUN_ID, run_id)
    if run_url:
        print_marker(MARKER_RUN_URL, run_url)
    
    print(f"\nJob output:\n{stdout}")
    
    # Poll for completion if we have a run_id
    if run_id:
        final_state = poll_run(run_id, env)
        print_marker(MARKER_FINAL_STATE, final_state)
        
        # Try to get notebook output
        fetch_run_output(run_id, env)
        
        if final_state not in ("SUCCEEDED", "SUCCESS"):
            print(f"\n✗ Job failed with state: {final_state}")
            sys.exit(1)
        else:
            print(f"\n✓ Job completed successfully!")
    else:
        print("\nNote: Could not extract run_id - check job status manually")
        if run_url:
            print(f"  URL: {run_url}")


def poll_run(run_id: str, env: dict) -> str:
    """Poll run until terminal state. Returns final state."""
    print(f"\nPolling run {run_id} for completion...")
    
    terminal_states = {
        "TERMINATED", "SKIPPED", "INTERNAL_ERROR",
        "SUCCESS", "FAILED", "CANCELED", "CANCELLED"
    }
    
    poll_interval = 10  # seconds
    max_polls = 360  # 1 hour max
    
    for i in range(max_polls):
        result = run_cmd(
            ["databricks", "runs", "get", "--run-id", run_id, "-o", "json"],
            capture=True,
            check=False,
            env_extra=env,
        )
        
        if result.returncode != 0:
            print(f"  Warning: Could not fetch run status")
            time.sleep(poll_interval)
            continue
        
        try:
            data = json.loads(result.stdout)
            state = data.get("state", {})
            life_cycle = state.get("life_cycle_state", "UNKNOWN")
            result_state = state.get("result_state", "")
            
            print(f"  [{i * poll_interval}s] State: {life_cycle} {result_state}".strip())
            
            if life_cycle in terminal_states or result_state in terminal_states:
                return result_state or life_cycle
                
        except json.JSONDecodeError:
            print(f"  Warning: Could not parse run status")
        
        time.sleep(poll_interval)
    
    return "TIMEOUT"


def fetch_run_output(run_id: str, env: dict) -> None:
    """Fetch and print notebook exit value."""
    print(f"\nFetching run output...")
    
    result = run_cmd(
        ["databricks", "runs", "get-output", "--run-id", run_id, "-o", "json"],
        capture=True,
        check=False,
        env_extra=env,
    )
    
    if result.returncode != 0:
        print("  Note: Could not fetch run output (may not be available)")
        return
    
    try:
        data = json.loads(result.stdout)
        notebook_output = data.get("notebook_output", {})
        exit_value = notebook_output.get("result", "")
        
        if exit_value:
            print(f"\nNotebook exit value:")
            print_marker(MARKER_RESULT_JSON, exit_value)
            
            # Try to pretty-print if it's JSON
            try:
                parsed = json.loads(exit_value)
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                print(exit_value)
        else:
            print("  No exit value returned from notebook")
            
    except json.JSONDecodeError:
        print("  Could not parse run output")


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run job in smoke mode (fast/cheap)."""
    args.run_mode = "smoke"
    cmd_run(args)


def cmd_destroy(args: argparse.Namespace) -> None:
    """Destroy deployed bundle resources."""
    env = get_profile_env(args.profile)
    
    print("⚠ This will destroy all bundle resources in the target workspace!")
    if not args.yes:
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    
    run_cmd(
        ["databricks", "bundle", "destroy", "-t", args.target, "--auto-approve"],
        env_extra=env,
    )
    print("\n✓ Bundle destroyed!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Databricks Asset Bundle dev CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Global args
    parser.add_argument(
        "--profile", "-p",
        help="Databricks CLI profile name (or set DATABRICKS_CONFIG_PROFILE)",
    )
    parser.add_argument(
        "--target", "-t",
        default="dev",
        help="Bundle target (default: dev)",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # setup
    subparsers.add_parser("setup", help="Check prerequisites")
    
    # unit
    unit_parser = subparsers.add_parser("unit", help="Run local unit tests")
    unit_parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    
    # validate
    subparsers.add_parser("validate", help="Validate bundle config")
    
    # deploy
    subparsers.add_parser("deploy", help="Deploy bundle")
    
    # run
    run_parser = subparsers.add_parser("run", help="Run the job")
    run_parser.add_argument("--env", help="Environment override")
    run_parser.add_argument(
        "--run-mode",
        default="full",
        choices=["full", "smoke"],
        help="Run mode (default: full)",
    )
    run_parser.add_argument(
        "--json-params",
        default="{}",
        help='Extra params as JSON string (default: "{}")',
    )
    
    # smoke
    smoke_parser = subparsers.add_parser("smoke", help="Run job in smoke mode")
    smoke_parser.add_argument("--env", help="Environment override")
    smoke_parser.add_argument(
        "--json-params",
        default="{}",
        help='Extra params as JSON string',
    )
    
    # destroy
    destroy_parser = subparsers.add_parser("destroy", help="Destroy bundle resources")
    destroy_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    # Dispatch
    commands = {
        "setup": cmd_setup,
        "unit": cmd_unit,
        "validate": cmd_validate,
        "deploy": cmd_deploy,
        "run": cmd_run,
        "smoke": cmd_smoke,
        "destroy": cmd_destroy,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
