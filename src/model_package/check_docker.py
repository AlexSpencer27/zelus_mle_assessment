import subprocess
import sys
import os
from pathlib import Path

log_folder = Path(__file__).parents[2] / "data" / "docker"
os.makedirs(log_folder, exist_ok=True)
LOG_FILE = Path(os.path.join(log_folder, "check_docker.log"))


def check_docker():
    """Check if Docker daemon is running."""
    try:
        # Test Docker with `docker info`
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        LOG_FILE.write_text("Docker daemon is running.\n")
        print("Docker daemon is running.")
        sys.exit(0)
    except subprocess.CalledProcessError:
        LOG_FILE.write_text(
            "Docker daemon is not running. Please start Docker and try again.\n"
        )
        print(
            "Docker daemon is not running. Please start Docker and try again.",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError:
        LOG_FILE.write_text(
            "Docker is not installed or not in PATH. Please install Docker to proceed.\n"
        )
        print(
            "Docker is not installed or not in PATH. Please install Docker to proceed.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    check_docker()
