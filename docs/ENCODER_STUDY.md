# Encoder study: does CAIP transfer to a parallel gripper?

## Question

CAIP (Sharma et al., 2026, [arXiv 2606.17256](https://arxiv.org/abs/2606.17256)) continues training SigLIP 2 with a contrastive loss that aligns text-conditioned image embeddings with future human hand motion from about 32,000 hours of egocentric video. On a 22-DoF dexterous-hand robot it reaches 76% average success against 43% for SigLIP 2. The paper lists transfer to grippers as untested, since the hand-pose proxy may bias features toward human hands.

This study asks whether those features still help a 6-DoF arm with a two-finger gripper, trained from about 50 teleoperated demos per task.

## Design

| | Choice | Why |
|---|---|---|
| Encoders | SigLIP 2 ViT-L/16-256, DINOv2 ViT-L/14, CAIP ViT-L/16-256 | Same size and a 16×16 patch grid; SigLIP 2 is CAIP's initialization, so SigLIP 2 vs CAIP isolates the action pre-training |
| Encoders frozen | Yes | Measures the representation, not fine-tuning; features can be cached once |
| Feature kinds | `pooled` (1 token per camera), `grid` (4×4 = 16 tokens per camera), `masked` (object-mask mean + pooled) | Global vs spatial vs object-centric |
| Policy head | 4-layer transformer encoder + 4-layer decoder with 50 action queries, L1 loss (ACT without the CVAE) | Identical for every encoder |
| Task conditioning | Learned task embedding, same for all encoders | Fair comparison; CAIP's pooled feature is also text-conditioned by design |
| Cameras | wrist + front, 640×480, square-cropped before encoding | Square crop keeps patches aligned with masks |
| Data | Same LeRobotDataset, split by episode (90/10) | Validation L1 picks the checkpoint |
| Evaluation | Real robot, 4×4 grid × {nominal, dim_light, distractors}, Wilson 95% intervals | Mirrors CAIP's robustness axes |

## Run it

```powershell
.\scripts\run_encoder_study.ps1 -Dataset kayarecepomer/so101_pick_place
.\scripts\run_encoder_study.ps1 -Dataset kayarecepomer/so101_pick_place -MaskQuery "toy" -Encoders caip,siglip2
```

Feature caching takes the longest; every later training run reuses the cache under `features/`. Each run writes `outputs/study/<dataset>_<encoder>_<kind>/best.pt` and `metrics.csv`.

Then for each checkpoint, serve it and evaluate on the real arm:

```powershell
python -m bridge.windows.policy_server --policy study --path outputs/study/so101_pick_place_caip_pooled/best.pt
```

```bash
python -m vla.eval.run_trials --label caip_pooled --task "Put the toy in the bowl" \
  --server tcp://<windows-ip>:5556 --conditions nominal dim_light distractors \
  --log logs/eval/study.jsonl
```

## Reporting

- The results table from `python -m vla.eval.summarize logs/eval/study.jsonl`.
- Validation L1 per run from `metrics.csv` (offline metric; report it next to real success, not instead of it).
- Two or three failure clips per encoder.
- If CAIP does not help, say so and show where it fails. A clear negative result on grippers is useful to the CAIP authors too.

## Notes

- Masks come from OWLv2 (box from a text query) and SAM (box to mask). If the object is not detected, the masked feature falls back to the mean of all patches, and the per-frame coverage is cached as `<camera>_coverage.npy` for checking how often that happens.
- CAIP loads through `transformers` with `trust_remote_code=True` from `yuvansharma/caip-vitl256` (MIT license per its model card). Read the remote code before running it.
