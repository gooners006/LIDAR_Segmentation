<#
.SYNOPSIS
  Completion input-gate ablation driver (Finding #27 hardening, all 11 sequences).

.DESCRIPTION
  For each labelled SemanticKITTI sequence, render a full-sequence GATE-OFF
  completion pass (the input gate disabled so EVERY car track completes, incl.
  fragments/merges). The gate-ON result is derived post-hoc in
  scratchpad/gate_ablation_analyze.py as the footprint-passing subset
  (fit_length >= 2.7 and fit_width <= 2.3), recorded per track as
  ref_fit_length/ref_fit_width. One render per sequence therefore yields both
  arms of the ablation; the frozen output/08 (shipped gate-ON) is the
  independent cross-check for the post-hoc reconstruction.

  Freeze-safe: writes only under output/experiments/gate_ablation_v2/; no edits
  to frozen artifacts. Resumable: a sequence whose tracks.json already exists is
  skipped.

.PARAMETER Seqs
  Sequences to render. Default: all 11 labelled sequences 00-10.

.PARAMETER Frames
  Max frames per render. Default 5000 (>= longest labelled sequence).

.EXAMPLE
  ./scripts/run_gate_ablation.ps1
  ./scripts/run_gate_ablation.ps1 -Seqs 04            # smoke test (271 frames)
#>
param(
    [string[]] $Seqs = @("00","01","02","03","04","05","06","07","08","09","10"),
    [int] $Frames = 5000
)

$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$outRoot = "output/experiments/gate_ablation_v2"
New-Item -ItemType Directory -Force (Join-Path $repo $outRoot) | Out-Null

$t0 = Get-Date
foreach ($seq in $Seqs) {
    $tracksJson = Join-Path $repo "$outRoot/$seq/tracks.json"
    Write-Host "==================== SEQ $seq (gate OFF) ====================" -ForegroundColor Cyan

    if (Test-Path $tracksJson) {
        Write-Host "[$seq] tracks.json exists - skipping" -ForegroundColor Yellow
        continue
    }

    $log = Join-Path $repo "$outRoot/${seq}_render.log"
    $tSeq = Get-Date
    & $py src/main.py --seq $seq --frames $Frames --no-gui --save-output `
        --no-gate --out-root $outRoot *> $log
    if ($LASTEXITCODE -ne 0) { throw "[$seq] render failed (exit $LASTEXITCODE); see $log" }
    Write-Host ("[$seq] done in {0:n1} min" -f ((Get-Date) - $tSeq).TotalMinutes)
}

Write-Host ("`nGate-ablation renders complete in {0:n1} min total. Analyze with:" -f ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green
Write-Host "  $py scratchpad/gate_ablation_analyze.py"
