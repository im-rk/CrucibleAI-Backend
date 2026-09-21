"""Run the real research pipeline from the command line and print the report.

  python scripts/run_research.py --query "Find UV LED materials with bandgap above 3.4 eV"
  python scripts/run_research.py --query "..." --json report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

sys.path.append("src")

from matsci_agent.api.dependencies import get_research_service  # noqa: E402
from matsci_agent.core.logging import configure_logging  # noqa: E402
from matsci_agent.domain.schemas import ResearchRequest  # noqa: E402


async def main(query: str, max_results: int, json_path: str | None) -> None:
    service = get_research_service()
    report = None
    async for event in service.stream(ResearchRequest(query=query, max_results=max_results)):
        name = event["event"]
        if name == "complete":
            report = event["report"]
        elif name == "error":
            sys.exit(f"Research failed: {event['message']}")
        elif name not in ("agent_complete",):
            detail = {k: v for k, v in event.items() if k not in ("event", "report")}
            print(f"  [{name}] {json.dumps(detail, default=str)[:160]}")

    print(f"\nMode: {report['mode']}  |  {report['duration_ms'] / 1000:.1f}s  |  grounding: {report['grounding'].get('method')}")
    print(f"\nCriteria ({len(report['criteria'])}):")
    for c in report["criteria"]:
        print(f"  - {c['label']}")
    print(f"\nCandidates ({len(report['candidates'])}):")
    for c in report["candidates"]:
        print(f"  #{c['rank']} {c['formula'] or c['name']:<16} {c['status']:<10} score={c['score']}  flags={c['flags']}")
    print("\nHypotheses:")
    for h in report["hypotheses"]:
        print(f"  * [{h['confidence']}/{h['origin']}] {h['statement']}\n      test: {h['suggested_experiment']}")
    print("\nKnowledge gaps:")
    for g in report["knowledge_gaps"]:
        print(f"  - {g}")
    print(f"\nCitations: {len(report['citations'])}   Warnings:")
    for w in report["warnings"]:
        print(f"  ! {w}")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nFull report written to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", default="Find UV LED materials with bandgap above 3.4 eV")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()
    configure_logging()
    asyncio.run(main(args.query, args.max_results, args.json_path))
