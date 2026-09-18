# Session Handoff / Project State

This file is for the next Claude Code session working in this repo. Read this first, then [README.md](README.md) (project overview/showcase) and [SO101_System_Architecture_Handover.md](SO101_System_Architecture_Handover.md) (full hardware specs, calibration values, long-term research architecture) for the rest of the picture.

## What this repo is

Windows/simulation half of a two-machine SO-101 dual-arm robotics research platform. macOS runs the physical hardware (LeRobot, real arms) — not part of this repo. This machine runs NVIDIA Isaac Sim and will eventually run policy training. See README.md for the full pitch; this file is the terser "what's actually been done and what's next" version.

## Environment facts (this machine)

- **Isaac Sim**: built from source at `C:\Users\recep\isaacsim` (v6.0.1-rc.7). The actual runnable build is under `C:\Users\recep\isaacsim\_build\windows-x86_64\release\`:
  - `isaac-sim.bat` — full GUI
  - `python.bat` — bundled Python 3.12.13 (has `pyzmq` installed already)
  - `isaac-sim.compatibility_check.bat` — fast GPU/driver check, no full GUI
  - Compatibility check **passed**: RTX 5070, 12.82GB VRAM, driver 610.74, all above minimums.
- **No `gh`, `claude`, `pdflatex`/`xelatex`, or system `pip` on PATH.** `git` and `uv` are available. Use Isaac Sim's `python.bat -m pip install ...` for anything that needs to run inside the Isaac Sim Python env.
- **Gotcha learned the hard way**: launching a GUI app (Isaac Sim) via the Bash tool's `run_in_background` can report "completed" prematurely — the wrapping `.bat` exits after handing off to `kit.exe`, but the actual app is often still running fine. Don't trust that notification for GUI apps; verify with PowerShell (`Get-Process | Where ProcessName -match "kit"`, check `MainWindowTitle`). For anything that must outlive a single tool call cleanly, use PowerShell `Start-Process` (detached) with `dangerouslyDisableSandbox: true` rather than Bash `run_in_background`.

## What's built so far

- **Repo scaffold**: `sim/`, `bridge/mac/`, `bridge/windows/`, `scripts/`, `media/`, `resume/`.
- **ZeroMQ bridge** (verified end-to-end locally, not yet tested with the real Mac hardware):
  - `bridge/joint_state.py` — shared `JointState` dataclass (6 joints + timestamp), binary (`struct`) and JSON codecs.
  - `bridge/mac/publisher.py` — reads the leader arm via LeRobot's `FeetechMotorsBus`, publishes on `tcp://0.0.0.0:5555` at 60Hz. Written here, only runnable on the Mac.
  - `bridge/windows/subscriber.py` — subscribes, currently just logs received state (not yet wired to drive the Isaac Sim twin).
  - `scripts/fake_publisher.py` — local dummy publisher for testing the Windows side without the Mac. Round-trip encode/decode confirmed working.
- **Isaac Sim MCP server** (set up 2026-09-18, needs a session reload to activate — see below):
  - Cloned `whats2000/isaacsim-mcp-server` to `C:\Users\recep\isaacsim-mcp-server` (sibling of the Isaac Sim checkout, not inside this repo — it's a dev tool, not project source).
  - `uv sync` done there, venv built, `isaacsim-mcp-server.exe` console script present.
  - Isaac Sim is currently running via that repo's `scripts\run_isaac_sim.ps1` (which passes `--ext-folder`/`--enable` to load `isaac.sim.mcp_extension`), with `ISAACSIM_ROOT=C:\Users\recep\isaacsim\_build\windows-x86_64\release`. Extension confirmed listening on `127.0.0.1:8766`.
  - `.mcp.json` at this repo's root registers the `isaac-sim` MCP server for Claude Code (runs `isaacsim-mcp-server\scripts\run_mcp_server.ps1`). **Not committed yet** — it hardcodes an absolute machine-specific path; ask the user whether to commit it or gitignore it.
  - **MCP servers load at session start, so a session started before this file existed won't have the `isaac-sim` tools.** If you don't see Isaac Sim MCP tools available, tell the user to reload/restart the Claude Code window.
- **Resume side-quest** (unrelated to the robotics work, lives in `resume/`): two NVIDIA internship resumes (`resume_nvidia_swe.tex`, `resume_nvidia_dlca.tex`) tailored from the user's base resume, swapping in the SO-101 project. Repo is confirmed **private** by the user, which is why these (with phone/email) were pushed. Don't assume this is public-facing content like the README/media.
- **README.md** rewritten as a project showcase for a hiring-manager audience: build photos in `media/` (GPS EXIF stripped before commit), architecture diagram, status checklist, tech stack, future-work section (sim-to-real, VR teleoperation, agentic planning layer).

## Not done yet / open threads

- `sim/load_so101.py` doesn't exist yet — the SO-101 URDF/USD asset still needs to be found or converted and imported into Isaac Sim. This was the planned next step before the MCP detour; with the MCP server now available, this is a good candidate to do *through* the MCP tools instead of hand-written Isaac Sim API scripts.
- `bridge/windows/subscriber.py` only logs — not yet wired to actually move the Isaac Sim twin.
- Live end-to-end bridge test (real leader arm on Mac → Isaac Sim twin on Windows) has never been run — needs a Mac-side session to run `bridge/mac/publisher.py` and this machine to be listening.
- Phase 2 from the original plan (validating teleop + dataset recording on the Mac) was never executed from this session — it can't be, this session only has Windows access. Whoever picks this up on the Mac side should run the `lerobot_teleoperate.py` command documented in the handover doc §4.1.
- Repo was renamed on GitHub from `robot-sim-soarm101` to `vla-robotics-platform` (the old name read too much like a copy of the open-source SO-ARM101 project). The local clone directory is still named `robot-sim-soarm101` on disk — only the GitHub remote changed.
- Full original phased plan (if useful for context on sequencing decisions): `C:\Users\recep\.claude\plans\resilient-scribbling-origami.md` (local to this machine, not in the repo).

## Suggested next step

Reload the Claude Code session so the `isaac-sim` MCP tools activate, confirm they're available (e.g. ask it to list objects in the current Isaac Sim stage), then use them to import/build the SO-101 arm in the scene rather than hand-writing the asset-import script from scratch.
