"""Print a Markdown results table (success rate, 95% Wilson interval) from trial logs.

python -m vla.eval.summarize logs/eval/*.jsonl
"""

from __future__ import annotations

import argparse

from vla.eval.protocol import load_records, summarize, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real-robot evaluation logs.")
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    print(to_markdown(summarize(load_records(*args.logs))))


if __name__ == "__main__":
    main()
