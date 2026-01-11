# Repository-Proposal Alignment Verification

Write-Host "Checking Repository Alignment with Proposal..." -ForegroundColor Cyan
Write-Host "=" * 60

$all_aligned = $true

# Check RQ1: Federated Learning
Write-Host "`nRQ1: Formally Verified Federated Learning for SCION" -ForegroundColor Yellow
$rq1_items = @(
    "src\federated-learning\protocols\fedavg.py",
    "src\federated-learning\protocols\krum.py",
    "src\federated-learning\privacy\differential_privacy.py",
    "src\federated-learning\models\scion_features.py",
    "src\formal-verification\isabelle\federated_protocols\FedAvg_Convergence.thy"
)

foreach ($item in $rq1_items) {
    if (Test-Path $item) {
        Write-Host "  ✓ $item" -ForegroundColor Green
    } else {
        Write-Host "  ✗ MISSING: $item" -ForegroundColor Red
        $all_aligned = $false
    }
}

# Check RQ2: Zero-Knowledge
Write-Host "`nRQ2: Zero-Knowledge Privacy for Bandwidth Markets" -ForegroundColor Yellow
$rq2_items = @(
    "src\zero-knowledge\circuits\bandwidth_market.circom",
    "src\zero-knowledge\README.md"
)

foreach ($item in $rq2_items) {
    if (Test-Path $item) {
        Write-Host "  ✓ $item" -ForegroundColor Green
    } else {
        Write-Host "  ✗ MISSING: $item" -ForegroundColor Red
        $all_aligned = $false
    }
}

# Check RQ3: Moving Target Defense
Write-Host "`nRQ3: Moving Target Defense" -ForegroundColor Yellow
if (Test-Path "src\moving-target-defense\README.md") {
    Write-Host "  ✓ src\moving-target-defense\README.md" -ForegroundColor Green
} else {
    Write-Host "  ✗ MISSING: MTD documentation" -ForegroundColor Red
    $all_aligned = $false
}

# Check RQ4: Digital Twin
Write-Host "`nRQ4: Federated Digital Twin" -ForegroundColor Yellow
if (Test-Path "src\digital-twin\README.md") {
    Write-Host "  ✓ src\digital-twin\README.md" -ForegroundColor Green
} else {
    Write-Host "  ✗ MISSING: Digital Twin documentation" -ForegroundColor Red
    $all_aligned = $false
}

# Check RQ5: IoT-SCION
Write-Host "`nRQ5: Lightweight IoT-SCION Integration" -ForegroundColor Yellow
if (Test-Path "src\iot-scion\README.md") {
    Write-Host "  ✓ src\iot-scion\README.md" -ForegroundColor Green
} else {
    Write-Host "  ✗ MISSING: IoT-SCION documentation" -ForegroundColor Red
    $all_aligned = $false
}

# Check Supporting Infrastructure
Write-Host "`nSupporting Infrastructure" -ForegroundColor Yellow
$infrastructure = @(
    "experiments\README.md",
    "data\README.md",
    "docs\README.md",
    "tests\test_federated.py",
    ".github\workflows\tests.yml"
)

foreach ($item in $infrastructure) {
    if (Test-Path $item) {
        Write-Host "  ✓ $item" -ForegroundColor Green
    } else {
        Write-Host "  ✗ MISSING: $item" -ForegroundColor Red
        $all_aligned = $false
    }
}

Write-Host "`n" + ("=" * 60)
if ($all_aligned) {
    Write-Host "✅ Repository is FULLY ALIGNED with proposal!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some items missing - see above" -ForegroundColor Yellow
}

Write-Host "`nGitHub Repository: https://github.com/hmshujaatzaheer/federated-scion-security-framework"
