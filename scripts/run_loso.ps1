<#
.SYNOPSIS
  Leave-one-sequence-out (LOSO) cross-validation driver for car detection.

.DESCRIPTION
  For each held-out SemanticKITTI sequence K:
    1. Train a Stage B classifier on all OTHER sequences (K excluded from
       training AND checkpoint selection; last-epoch checkpoint is used).
    2. Run full-sequence detection evaluation on K with that fold's classifier,
       writing per-fold metrics JSON to results/loso/fold_<K>.json.

  Training and eval for a fold are skipped if their outputs already exist, so
  the calibration fold (run manually first) is not repeated and the sweep is
  resumable after an interruption.

.PARAMETER Seqs
  Sequences to run as folds. Default: all 11 labelled sequences 00-10.

.PARAMETER Frames
  Max frames per eval. Default 5000 (>= longest sequence, i.e. full sequence).

.EXAMPLE
  ./scripts/run_loso.ps1
  ./scripts/run_loso.ps1 -Seqs 00,01,02
#>
param(
    [string[]] $Seqs = @("00","01","02","03","04","05","06","07","08","09","10"),
    [int] $Frames = 5000,
    # Centered contiguous window fraction for eval (keeps frames consecutive so
    # the track filter still works). 1.0 = full sequence (default; single windows
    # proved too region-biased -- recall swung 0.26-0.76 across folds).
    [double] $FrameFraction = 1.0
)

$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$resultsDir = Join-Path $repo "results\loso"
New-Item -ItemType Directory -Force $resultsDir | Out-Null

foreach ($seq in $Seqs) {
    $tag = "stage_b_fold$seq"
    $ckpt = "checkpoints\${tag}_last.pth"
    $trainLog = Join-Path $resultsDir "fold_${seq}_train.log"
    $evalJson = Join-Path $resultsDir "fold_${seq}.json"
    $evalLog  = Join-Path $resultsDir "fold_${seq}_eval.log"

    Write-Host "==================== FOLD $seq ====================" -ForegroundColor Cyan

    # --- Train (skip if checkpoint already present) ---
    if (Test-Path $ckpt) {
        Write-Host "[$seq] checkpoint exists ($ckpt) - skipping training" -ForegroundColor Yellow
    } else {
        Write-Host "[$seq] training classifier (held-out seq $seq)..."
        $tTrain = Get-Date
        & $py src/train_classifier.py --stage-b --no-pretrain `
            --held-out-seq $seq --tag $tag --epochs 15 *> $trainLog
        if ($LASTEXITCODE -ne 0) { throw "[$seq] training failed (exit $LASTEXITCODE); see $trainLog" }
        Write-Host ("[$seq] training done in {0:n1} min" -f ((Get-Date) - $tTrain).TotalMinutes)
    }
    if (-not (Test-Path $ckpt)) { throw "[$seq] expected checkpoint missing: $ckpt" }

    # --- Evaluate held-out sequence (skip if JSON already present) ---
    if (Test-Path $evalJson) {
        Write-Host "[$seq] eval JSON exists ($evalJson) - skipping eval" -ForegroundColor Yellow
    } else {
        Write-Host "[$seq] evaluating detection on held-out seq $seq..."
        $tEval = Get-Date
        & $py src/evaluate.py --seq $seq --frames $Frames `
            --frame-fraction $FrameFraction `
            --classifier-ckpt $ckpt --json-out $evalJson *> $evalLog
        if ($LASTEXITCODE -ne 0) { throw "[$seq] eval failed (exit $LASTEXITCODE); see $evalLog" }
        Write-Host ("[$seq] eval done in {0:n1} min" -f ((Get-Date) - $tEval).TotalMinutes)
    }
}

Write-Host "`nLOSO sweep complete. Aggregate with:" -ForegroundColor Green
Write-Host "  $py scripts/aggregate_loso.py"
