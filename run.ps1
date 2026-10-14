# One entry point for everything. Run from PowerShell in the repository root:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 <command> [extra args]
# Commands: check | setup | test | bench | train01 | train01py | train02 | export | notes | all
param([Parameter(Position = 0)][string]$Cmd = "help", [Parameter(ValueFromRemainingArguments = $true)]$Rest)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PSScriptRoot\python"
$env:PIP_CACHE_DIR = "$PSScriptRoot\.cache\pip"   # keep downloads off C:
$Venv = "$PSScriptRoot\.venv\Scripts\python.exe"
$Py = if (Test-Path $Venv) { $Venv } else { "python" }

function Step($title, [scriptblock]$body) {
  Write-Host "`n=== $title ===" -ForegroundColor Cyan
  $global:LASTEXITCODE = 0
  & $body
  if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "$title failed (exit $LASTEXITCODE)" }
}

switch ($Cmd) {
  "check" { Step "environment report -> ENVIRONMENT.md" { & $Py python\check_env.py } }

  "setup" {
    # A venv on D: that *sees* the existing global packages but installs anything new on D:.
    if (-not (Test-Path $Venv)) { Step "create .venv (system site packages)" { python -m venv --system-site-packages .venv } }
    $Py = $Venv
    # torch: install the CUDA wheel by default when an NVIDIA GPU exists and the venv's torch is missing/CPU-only.
    # Pick cu130 for drivers reporting CUDA >= 13.0, else cu126. Skip with: run.ps1 setup -NoCuda  (pass "-NoCuda" as extra arg)
    $hasCuda = (& $Py -c "import torch; print(int(torch.version.cuda is not None))" 2>$null)
    if ($Rest -notcontains "-NoCuda" -and $hasCuda -ne "1" -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
      $smi = (nvidia-smi | Out-String)
      $tag = if ($smi -match "CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)" -and [int]$Matches[1] -ge 13) { "cu130" } else { "cu126" }
      Step "install CUDA torch ($tag)" { & $Py -m pip install --upgrade torch --index-url "https://download.pytorch.org/whl/$tag" }
    }
    Step "install small dependencies" { & $Py -m pip install -r requirements.txt }
    Step "environment report" { & $Py python\check_env.py }
  }

  "test" {
    Step "JS reference traces" { node tools\make_trace.js python\tests\js_traces.json }
    Step "parity + env tests" { & $Py -m pytest python\tests -q }
  }

  "bench"   { Step "CPU vs CUDA benchmark" { & $Py python\bench_device.py @Rest } }

  "train01" {
    Step "tabular Q 12x12" { node tools\train_tabular.js --size 12 }
    Step "tabular Q 6x6" { node tools\train_tabular.js --size 6 }
  }

  "train01py" { Step "tabular Q in Python (must reproduce the JS numbers)" { & $Py python\train_tabular.py @Rest } }

  "train02" { Step "DQN (CNN on grid)" { & $Py python\train_dqn.py @Rest } }

  "export" {
    # no args: (re)publish tryouts 01-02 into models\<algo>\ + models
egistry.js (what the game's Agent dropdown reads).
    # with args: python\export_model.py ... (e.g. --ckpt models\_runs\X_best.pt --algo 03_x --name "..." --tryout 3)
    if ($Rest) { Step "export" { & $Py python\export_model.py @Rest } } else { Step "publish agents to the model registry" { & $Py python\publish_baselines.py } }
    Step "check JS inference matches PyTorch" { node tools\check_model.js }
  }

  "notes" { Step "build HTML + DOCX notes" { & $Py notes\build_notes.py @Rest } }

  "all" {
    & $PSCommandPath check
    & $PSCommandPath test
    & $PSCommandPath bench
    & $PSCommandPath train02
    & $PSCommandPath export
    & $PSCommandPath notes
  }

  default { Get-Content $PSCommandPath -TotalCount 4 | ForEach-Object { $_.TrimStart('# ') } }
}
