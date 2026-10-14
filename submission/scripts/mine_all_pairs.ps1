$python = ".venv\Scripts\python.exe"

# Train: seq 00-07, 09-10 (same split as classifier Stage B)
Write-Host "=== Mining TRAIN pairs (seq 00-07, 09-10) ===" -ForegroundColor Cyan
& $python src/mine_completion_pairs.py --seq 00 01 02 03 04 05 06 07 09 10 --frames 99999 --output data/real_pairs/train

# Val: seq 08 (held-out)
Write-Host "`n=== Mining VAL pairs (seq 08) ===" -ForegroundColor Cyan
& $python src/mine_completion_pairs.py --seq 08 --frames 99999 --output data/real_pairs/val

# Summary
Write-Host "`n=== Done ===" -ForegroundColor Green
foreach ($split in @("train", "val")) {
    $splitDir = "data\real_pairs\$split"
    if (Test-Path $splitDir) {
        $classes = Get-ChildItem $splitDir -Directory
        foreach ($cls in $classes) {
            $dense = (Get-ChildItem $cls.FullName -Filter "dense_*.npy").Count
            $sparse = (Get-ChildItem $cls.FullName -Filter "sparse_*.npy").Count
            Write-Host "$split/$($cls.Name): $dense tracks, $sparse sparse frames"
        }
    }
}
