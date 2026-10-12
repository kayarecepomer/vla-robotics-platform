# Runbook

Commands in the order you run them. "Mac" means the `lerobot` conda env on the Mac; "Windows" means a Python env with CUDA PyTorch and `requirements/windows.txt`; "Isaac" means `C:\Users\recep\isaacsim\_build\windows-x86_64\release\python.bat`. Run everything from the repo root after `pip install -e .`.

Both machines must be on the same network. Find the Windows IP with `ipconfig` and the Mac IP with `ipconfig getifaddr en0`, and allow inbound TCP 5555-5556 in Windows Defender Firewall for the Python executable.

## 0. One-time setup

Mac:

```bash
pip install -r requirements/mac.txt && pip install -e .
lerobot-find-cameras opencv          # note the indices of the wrist and front cameras
huggingface-cli login                # so datasets can be pushed to the Hub
```

Windows:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/windows.txt; pip install -e .
huggingface-cli login
```

All joints are recorded in degrees (`use_degrees=true`) so the dataset, the policy client and the Isaac Sim twin share one unit. If your LeRobot version rejects `--robot.use_degrees`, remove that flag everywhere and set every arm joint in `config/so101_twin_mapping.json` to the `percent` mapping instead.

## 1. Record a dataset (Mac)

Print the placement grid once and tape it in front of the robot, row A nearest the base:

```bash
python -m vla.eval.run_trials --print-grid grid.svg     # print at 100% scale
WRIST_CAM=0 FRONT_CAM=1 scripts/record_dataset.sh pick_place "Put the toy in the bowl" 50
```

Vary the object's start cell across episodes so every grid cell appears in training. For the goal-conditioned color task, record one session per color into the same dataset:

```bash
for color in red green blue; do
  scripts/record_dataset.sh color_blocks "Put the ${color} block in the bowl" 20 --resume=true
done
```

## 2. Train baselines (Windows)

```powershell
.\scripts\train_baselines.ps1 -Dataset kayarecepomer/so101_pick_place
```

## 3. Check latency, then run a policy on the real arm

Windows, then Mac:

```powershell
python -m bridge.windows.policy_server --policy hold
```

```bash
python -m bridge.mac.policy_client --server tcp://<windows-ip>:5556 --cameras wrist=0,front=1 --seconds 10
```

The `hold` policy only repeats the current joint state, so the arm stays still while the client prints round-trip latency and control rate. Then serve a real checkpoint:

```powershell
python -m bridge.windows.policy_server --policy lerobot --path outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model
```

```bash
python -m bridge.mac.policy_client --server tcp://<windows-ip>:5556 --task "Put the toy in the bowl"
```

## 4. Evaluate (Mac, with the server running)

```bash
python -m vla.eval.run_trials --label act --task "Put the toy in the bowl" \
  --server tcp://<windows-ip>:5556 --conditions nominal dim_light distractors \
  --log logs/eval/act_pick_place.jsonl
python -m vla.eval.summarize logs/eval/*.jsonl
```

Trials already in the log are skipped, so a session can be stopped with `q` and resumed later.

## 5. Real-to-sim mirror (Isaac + Mac)

```powershell
$IsaacPython = "C:\Users\recep\isaacsim\_build\windows-x86_64\release\python.bat"
& $IsaacPython sim\so101_scene.py --save sim\so101_scene.usd --headless   # optional: persist the stage
& $IsaacPython sim\mirror.py --connect tcp://<mac-ip>:5555
```

```bash
python -m bridge.mac.publisher                       # leader arm drives the twin
# or, while a policy runs:
python -m bridge.mac.policy_client --server tcp://<windows-ip>:5556 --publish tcp://0.0.0.0:5555
```

If a twin joint moves the wrong way, flip its `sign` in `config/so101_twin_mapping.json`; if it is off by a constant angle, set `offset_rad`. Replays need no Mac:

```bash
python scripts/export_episode.py --repo-id kayarecepomer/so101_pick_place --episode 0 --out episode_000.npz
```

```powershell
& $IsaacPython sim\mirror.py --replay episode_000.npz --loop
```

Screen-record the real arm and the twin side by side for the demo video.

## 6. DAgger (Mac, with the server running)

```bash
python -m vla.dagger --server tcp://<windows-ip>:5556 --dataset kayarecepomer/so101_pick_place --task "Put the toy in the bowl"
```

Type `t` + Enter to take over with the leader arm when the policy is about to fail, `t` again to hand back, `q` to quit and upload. Retrain on the grown dataset, then evaluate again under a new `--label`.

## 7. Encoder study (Windows)

See [ENCODER_STUDY.md](ENCODER_STUDY.md).
