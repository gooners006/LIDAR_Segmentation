# Export the built deck for the defense: a PDF backup and a PPTX with its fonts (Segoe UI) embedded,
# for a presentation laptop that may not have Segoe UI. Uses PowerPoint itself (COM), so the PDF
# matches what PowerPoint shows. Closes only the files it opens and never quits PowerPoint, so a
# deck you have open stays open.
#
# Run (after build_deck.py):  pwsh docs/defense/scripts/export_deck.ps1 [-Source deck_v2_new.pptx]
param([string]$Source = "deck_v2.pptx")

$dir = Split-Path -Parent $PSScriptRoot          # docs/defense
$src = Join-Path $dir $Source
$pp = New-Object -ComObject PowerPoint.Application

$pres = $pp.Presentations.Open($src, $true, $false, $false)                  # read-only, no window
$pres.SaveCopyAs((Join-Path $dir "deck_v2.pdf"), 32)                          # ppSaveAsPDF
$pres.Close()

$pres = $pp.Presentations.Open($src, $false, $false, $false)
$pres.SaveAs((Join-Path $dir "deck_v2_embedded.pptx"), 24, -1)                # OpenXML, embed TrueType fonts
$pres.Close()

Get-ChildItem (Join-Path $dir "deck_v2*") | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
