# robot-sim-soarm101

Windows/simulation half of a two-machine SO-101 dual-arm robotics research platform. Full context, hardware specs, calibration data, and the research roadmap are in [SO101_System_Architecture_Handover.md](SO101_System_Architecture_Handover.md).

## Machine split

- **macOS** — physical hardware node. Runs LeRobot, drives the real SO-101 leader/follower arms over USB serial, records teleoperation datasets. Not part of this repo.
- **Windows (this repo)** — simulation and training node. NVIDIA Isaac Sim digital twin, the Mac↔Windows communication bridge, and (eventually) policy training.

## Layout

- `sim/` — Isaac Sim scripts: asset import, joint control, scene setup.
- `bridge/mac/` — communication code that runs on the Mac (reads live arm state, publishes it). Written here, executed there.
- `bridge/windows/` — communication code that runs here (subscribes to arm state, drives the sim twin).
- `scripts/` — one-off utility and verification scripts.
