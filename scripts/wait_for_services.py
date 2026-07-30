from __future__ import annotations

import os
import time

import httpx


def main() -> None:
    timeout_seconds = float(os.getenv("STRATA_SERVICE_TIMEOUT_SECONDS", "180"))
    web_port = int(os.getenv("STRATA_WEB_PORT", "3000"))
    services = {
        "MLflow": "http://localhost:5000/health",
        "Prefect": "http://localhost:4200/api/health",
        "Research API": "http://localhost:8000/health",
        "Research console": f"http://localhost:{web_port}/",
    }
    pending = dict(services)
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(timeout=3, trust_env=False) as client:
        while pending and time.monotonic() < deadline:
            for name, url in list(pending.items()):
                try:
                    response = client.get(url)
                except httpx.HTTPError:
                    continue
                if response.is_success:
                    del pending[name]
                    print(f"{name} is ready")
            if pending:
                time.sleep(2)
    if pending:
        missing = ", ".join(sorted(pending))
        raise SystemExit(f"Services did not become healthy within the timeout: {missing}")


if __name__ == "__main__":
    main()
