# Train the Tier A baselines on the Windows GPU (RTX 5070, 12 GB). Run from the repo root
# in an environment with lerobot installed.
#
#   .\scripts\train_baselines.ps1 -Dataset kayarecepomer/so101_pick_place
#   .\scripts\train_baselines.ps1 -Dataset kayarecepomer/so101_pick_place -Only smolvla
param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [ValidateSet("all", "act", "smolvla")][string]$Only = "all"
)
$ErrorActionPreference = "Stop"
$Name = $Dataset.Split("/")[-1]

if ($Only -in @("all", "act")) {
    # ACT from scratch: ~100k steps at batch 8 fits in 12 GB.
    lerobot-train `
        --dataset.repo_id=$Dataset `
        --policy.type=act `
        --policy.device=cuda `
        --policy.push_to_hub=false `
        --batch_size=8 `
        --steps=100000 `
        --save_freq=20000 `
        --output_dir=outputs/train/act_$Name `
        --job_name=act_$Name `
        --wandb.enable=false
}

if ($Only -in @("all", "smolvla")) {
    # SmolVLA (~450M params) fine-tuned from the community-pretrained base checkpoint.
    lerobot-train `
        --policy.path=lerobot/smolvla_base `
        --dataset.repo_id=$Dataset `
        --policy.device=cuda `
        --policy.push_to_hub=false `
        --batch_size=16 `
        --steps=20000 `
        --save_freq=5000 `
        --output_dir=outputs/train/smolvla_$Name `
        --job_name=smolvla_$Name `
        --wandb.enable=false
}
