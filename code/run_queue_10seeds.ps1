# Runs the 10-seed controls (seeds 5-9) in order. Safe to re-run: every step skips finished runs.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File code\run_queue_10seeds.ps1
# Pause:  create a file named PAUSE in the repository root (stops after the current run)
# Resume: delete PAUSE and run this script again
$ErrorActionPreference = "Continue"   # warnings on stderr must not stop the queue; failures are caught by exit code
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "code"
$env:PYTHONUNBUFFERED = "1"
$log = ".\figs\queue10_log.txt"
New-Item -ItemType Directory -Force .\figs | Out-Null

function Step($name, [string[]]$cmd) {
    if (Test-Path .\PAUSE) { "[$(Get-Date -Format s)] paused before: $name" | Tee-Object -Append $log; exit 0 }
    "[$(Get-Date -Format s)] START $name" | Tee-Object -Append $log
    & python @cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append $log
    if ($LASTEXITCODE -ne 0) { "[$(Get-Date -Format s)] FAILED $name (exit $LASTEXITCODE)" | Tee-Object -Append $log; exit 1 }
    "[$(Get-Date -Format s)] END $name" | Tee-Object -Append $log
}

# 1. the runner must reproduce two stored runs exactly before anything else
Step "verify runner (seed 1)" @("code/run_mech10.py", "--seeds", "1", "--arms", "clean_random", "uncertainty", "--out-dir", "./figs/mech10_verify")
Step "compare with stored runs" @("code/check_mech10_reproduces.py")

# 2. mechanism controls, seeds 5-9
Step "mechanism controls s5-9" @("code/run_mech10.py", "--seeds", "5", "6", "7", "8", "9")

# 3. seed-pool controls, seeds 5-9 (same settings as seeds 0-4)
$seeds = @("--seeds", "5", "6", "7", "8", "9")
Step "seedpool bodmas 844" (@("code/run_seedpool.py", "--dataset", "bodmas", "--pool-size", "844", "--train-months", "2", "--rates", "0.01") + $seeds)
Step "seedpool bodmas 3000" (@("code/run_seedpool.py", "--dataset", "bodmas", "--pool-size", "3000", "--train-months", "2", "--rates", "0.01") + $seeds)
Step "seedpool bodmas full" (@("code/run_seedpool.py", "--dataset", "bodmas", "--pool-size", "0", "--train-months", "2", "--rates", "0.005", "0.01") + $seeds)
Step "seedpool ember 844" (@("code/run_seedpool.py", "--dataset", "ember", "--pool-size", "844", "--train-months", "1", "--rates", "0.001", "0.0025", "0.01") + $seeds)

"[$(Get-Date -Format s)] QUEUE COMPLETE" | Tee-Object -Append $log
