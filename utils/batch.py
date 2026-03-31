"""Trading System Batch Processor.
Runs pipeline: tests, portfolio analysis, PE ratios, asset screening, MyTickers plot.
Usage: python batch.py [--debug] [--nopause]
--nopause passes NOPAUSE=1 to child scripts to skip input prompts.
"""

import os
from pathlib import Path
import subprocess
import sys

from config import SCRIPT_TIMEOUT_SECONDS

import platform


def is_debug_mode() -> bool:
    """Check if running in debug mode using environment variable or command line flag."""
    return (
        sys.gettrace() is not None
        or 'debugpy' in sys.modules
        or os.environ.get('DEBUG_MODE', '').lower() in ('1', 'true', 'yes')
        or '--debug' in sys.argv
    )


def is_nopause_mode() -> bool:
    """Check if --nopause flag is set to skip input confirmations."""
    return '--nopause' in sys.argv


def clear_screen() -> None:
    """Clear terminal screen cross-platform."""
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def run_script(script_path: str, description: str = "") -> bool:
    """Run a Python script as a subprocess.
    Args:
        script_path: Path to the Python script to run
        description: Human-readable description for logging
    Returns:
        True if script completed successfully, False otherwise
    Side Effects:
        - Spawns subprocess running script_path
        - Prints progress and errors to stdout
    """
    try:
        print(f"Running {script_path}{' - ' + description if description else ''}...")
        project_root = Path(__file__).parent.parent
        env = os.environ.copy()
        if is_nopause_mode():
            env['NOPAUSE'] = '1'
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=project_root,
            env=env,
            timeout=SCRIPT_TIMEOUT_SECONDS,
        )
        print(f"Completed {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: return code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Timeout running {script_path} after {SCRIPT_TIMEOUT_SECONDS}s")
        return False
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return False


if __name__ == "__main__":
    if not is_debug_mode() and not is_nopause_mode():
        print("Running in production mode. Use --debug, --nopause, or set DEBUG_MODE=1 to skip confirmation.")
        if input("Continue? Press 'y' to proceed: ").lower() != 'y':
            print("Exiting.")
            sys.exit(0)
    elif is_debug_mode():
        print("Running in debug mode - skipping confirmation.")
    elif is_nopause_mode():
        print("Running in nopause mode - skipping confirmation.")

    clear_screen()
    print("=" * 50)
    print("Running trading-system batch processes")
    print("=" * 50)

    scripts = [
        (None, 'Test suite'),
        ('Monitor/portfolio_performance.py', 'Portfolio performance analysis'),
        ('Analyze/plot_annualized_returns_grid.py', 'Annualized returns grid plot'),
        ('Analyze/myPE.py', 'PE ratios analysis'),
        ('Select_assets/screen.py', 'Asset screening'),
        ('Analyze/plot_MyTickers.py', 'MyTickers plot'),
    ]

    failed_scripts = []
    for script_path, description in scripts:
        if script_path is None:
            try:
                print(f"Running pytest - {description}...")
                project_root = Path(__file__).parent.parent
                env = os.environ.copy()
                if is_nopause_mode():
                    env['NOPAUSE'] = '1'
                result = subprocess.run(
                    [sys.executable, '-m', 'pytest', 'tests/', '-v'],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    env=env,
                )
                if result.stdout:
                    print(result.stdout.strip())
            except subprocess.CalledProcessError as e:
                print(f"Error running pytest: return code {e.returncode}")
                if e.stdout:
                    print(e.stdout)
                if e.stderr:
                    print(e.stderr)
                failed_scripts.append('pytest')
        else:
            if not run_script(script_path, description):
                failed_scripts.append(script_path)

    print("\n" + "=" * 50)
    if failed_scripts:
        print(f"Batch completed with {len(failed_scripts)} failed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")
    else:
        print("All scripts completed successfully!")
    print("=" * 50)
