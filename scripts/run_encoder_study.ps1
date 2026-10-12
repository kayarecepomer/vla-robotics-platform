# Encoder study on the Windows GPU: cache features for each frozen encoder, then train the
# same policy head on each. Afterwards, evaluate every best.pt on the real arm with
# bridge.windows.policy_server (--policy study) and vla.eval.run_trials.
#
#   .\scripts\run_encoder_study.ps1 -Dataset kayarecepomer/so101_pick_place
#   .\scripts\run_encoder_study.ps1 -Dataset kayarecepomer/so101_pick_place -MaskQuery "toy"
param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [string[]]$Encoders = @("siglip2", "dinov2", "caip"),
    [string[]]$Kinds = @("pooled", "grid"),
    [string]$MaskQuery = "",
    [int]$Steps = 20000
)
$ErrorActionPreference = "Stop"
$Name = $Dataset.Split("/")[-1]

foreach ($Encoder in $Encoders) {
    $FeatureDir = "features/$Name/$Encoder"
    if (-not (Test-Path "$FeatureDir/meta.json")) {
        $Extra = @()
        if ($MaskQuery) { $Extra = @("--mask-query", $MaskQuery) }
        python -m vla.study.extract_features --repo-id $Dataset --encoder $Encoder --out $FeatureDir @Extra
    }
    $RunKinds = $Kinds
    if ($MaskQuery) { $RunKinds = $Kinds + @("masked") }
    foreach ($Kind in $RunKinds) {
        python -m vla.study.train_head --features $FeatureDir --kind $Kind --steps $Steps
    }
}
Get-ChildItem outputs/study -Filter best.pt -Recurse | ForEach-Object { $_.FullName }
