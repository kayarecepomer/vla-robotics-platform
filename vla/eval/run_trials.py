"""Operator-driven real-robot evaluation (run on the Mac while the policy server runs on Windows).

    python -m vla.eval.run_trials --label act --task "Put the toy in the bowl" \
        --server tcp://<windows-ip>:5556 --conditions nominal dim_light distractors \
        --log logs/eval/act_pick_place.jsonl

For each scheduled trial the script tells you where to place the object, runs the policy for
up to ``--seconds`` (Ctrl+C ends a trial early), and asks whether it succeeded. Re-running with
the same ``--log`` skips trials already recorded, so a session can be split across days.
``python -m vla.eval.run_trials --print-grid grid.svg`` writes the printable placement grid.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from bridge.joint_state import JOINT_NAMES
from vla.eval.protocol import CONDITIONS, Grid, TrialRecord, append_record, grid_sheet_svg, load_records, trial_schedule


def ask(prompt: str, choices: str) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer


def move_to(robot, target: np.ndarray, seconds: float = 2.0, fps: float = 30.0) -> None:
    from bridge.lerobot_compat import joint_action

    reading = robot.get_observation()
    start = np.asarray([reading[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    steps = max(1, int(seconds * fps))
    for step in range(1, steps + 1):
        robot.send_action(joint_action(start + (target - start) * step / steps))
        time.sleep(1.0 / fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run logged real-robot evaluation trials.")
    parser.add_argument("--print-grid", help="Write the printable grid SVG to this path and exit")
    parser.add_argument("--label", help="Name for this policy in the results table, e.g. act or caip_pooled")
    parser.add_argument("--task", default="")
    parser.add_argument("--server", default="tcp://localhost:5556")
    parser.add_argument("--cameras", default="wrist=0,front=1")
    parser.add_argument("--conditions", nargs="+", default=["nominal"], choices=sorted(CONDITIONS))
    parser.add_argument("--repeats", type=int, default=1, help="Trials per cell per condition")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--execute-steps", type=int, default=20)
    parser.add_argument("--home", help="Comma-separated joint targets (degrees) to return to before each trial")
    parser.add_argument("--log", default="logs/eval/trials.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    grid = Grid()
    if args.print_grid:
        Path(args.print_grid).write_text(grid_sheet_svg(grid))
        print(f"Wrote {args.print_grid}. Print it at 100% scale.")
        return
    if not args.label:
        parser.error("--label is required to run trials")

    from bridge.lerobot_compat import make_follower
    from bridge.mac.policy_client import PolicyClient, run_episode

    done = set()
    if Path(args.log).exists():
        done = {record.key for record in load_records(args.log) if record.label == args.label}
    schedule = [slot for slot in trial_schedule(grid, args.conditions, args.repeats, args.seed) if slot.key not in done]
    print(f"{len(schedule)} trials to run ({len(done)} already logged in {args.log}).")

    robot = make_follower(cameras=args.cameras)
    robot.connect()
    client = PolicyClient(args.server)
    home = np.asarray([float(value) for value in args.home.split(",")], dtype=np.float32) if args.home else None
    current_condition = None
    try:
        for index, slot in enumerate(schedule, start=1):
            if slot.condition != current_condition:
                current_condition = slot.condition
                input(f"\n=== Condition '{slot.condition}': {CONDITIONS[slot.condition]}\nSet it up, then press Enter.")
            if home is not None:
                move_to(robot, home)
            x_cm, y_cm = grid.center_cm(slot.cell)
            action = ask(
                f"[{index}/{len(schedule)}] Place the object on cell {slot.cell} ({x_cm:.0f} cm, {y_cm:.0f} cm). "
                "Enter = run, s = skip, q = quit: ",
                {"", "s", "q"},
            )
            if action == "q":
                break
            if action == "s":
                continue
            stats = run_episode(robot, client, args.task, execute_steps=args.execute_steps, max_seconds=args.seconds)
            outcome = ask("Result? y = success, g = grasped but failed, n = failed: ", {"y", "g", "n"})
            notes = input("Notes (optional): ").strip()
            append_record(
                args.log,
                TrialRecord(
                    label=args.label,
                    task=args.task,
                    condition=slot.condition,
                    cell=slot.cell,
                    repeat=slot.repeat,
                    success=outcome == "y",
                    grasped=outcome in {"y", "g"},
                    duration_s=stats.duration_s,
                    latency=stats.summary(),
                    notes=notes,
                ),
            )
    finally:
        client.close()
        robot.disconnect()
    print(f"Done. Summarize with: python -m vla.eval.summarize {args.log}")


if __name__ == "__main__":
    main()
