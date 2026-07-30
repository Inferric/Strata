#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from strata_ot.autonomy.contracts import load_and_validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a bounded research proposal")
    parser.add_argument("proposal", type=Path)
    args = parser.parse_args()
    proposal = load_and_validate(args.proposal)
    print(f"Valid proposal: {proposal['proposal_id']}")


if __name__ == "__main__":
    main()
