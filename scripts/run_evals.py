"""Run the evaluation suites and write evaluation/results/latest.json and docs/EVALUATION.md.

    python scripts/run_evals.py --suite retrieval,extraction            # fast, no LLM needed
    python scripts/run_evals.py --suite injection                       # needs an LLM key
    python scripts/run_evals.py --suite e2e --e2e-llm --cases 2         # whole pipeline, slow with an LLM
    python scripts/run_evals.py --suite all                             # everything the configured LLM allows

Results from separate runs are merged, so running one suite never erases the others.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.append("src")

from matsci_agent.api.dependencies import get_llm_client  # noqa: E402
from matsci_agent.evaluation import suite  # noqa: E402

ALL = {"retrieval", "extraction", "injection", "e2e"}
DOC = Path("docs/EVALUATION.md")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="retrieval,extraction", help="comma-separated: retrieval, extraction, injection, e2e, all")
    parser.add_argument("--e2e-llm", action="store_true", help="also run the end-to-end cases with the LLM (slow)")
    parser.add_argument("--cases", type=int, default=None, help="limit the end-to-end cases")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    chosen = ALL if args.suite == "all" else {s.strip() for s in args.suite.split(",")}
    if not chosen <= ALL:
        sys.exit(f"Unknown suite(s): {sorted(chosen - ALL)}. Choose from {sorted(ALL)}.")

    llm = get_llm_client()
    print(f"LLM: {llm.model if llm.available else 'none (' + str(llm.unavailable_reason) + ')'}")
    results = await suite.run_suite(chosen, llm, e2e_cases=args.cases, e2e_llm=args.e2e_llm)
    suite.save(results)
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text(suite.render_markdown(suite.load()), encoding="utf-8")
    print(f"Wrote {suite.RESULTS} and {DOC}")
    # An invariant violation is always a bug (see evaluation/e2e_eval.py), so it fails the run (and CI).
    bad = [
        f"{name}: {case['goal'][:60]}: {[c['name'] for c in case['failed_invariants']]}"
        for name in ("e2e_heuristic", "e2e_llm") for case in results.get(name, {}).get("cases", []) if case["failed_invariants"]
    ]
    if bad:
        sys.exit("Invariant violations: " + "; ".join(bad))


if __name__ == "__main__":
    asyncio.run(main())
