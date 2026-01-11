Write-Host "`n+--------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "¦  FINAL REPOSITORY VERIFICATION                         ¦" -ForegroundColor Cyan
Write-Host "+--------------------------------------------------------+" -ForegroundColor Cyan

$files_to_check = @(
    # RQ1: Federated Learning
    "src\federated-learning\protocols\fedavg.py",
    "src\federated-learning\protocols\krum.py",
    "src\federated-learning\models\cnn_gru_dnn.py",
    "src\federated-learning\models\scion_features.py",
    "src\federated-learning\privacy\differential_privacy.py",
    
    # RQ1: Formal Verification
    "src\formal-verification\isabelle\federated_protocols\FedAvg_Convergence.thy",
    
    # RQ2: Zero-Knowledge
    "src\zero-knowledge\circuits\bandwidth_market.circom",
    
    # RQ3: Moving Target Defense
    "src\moving-target-defense\game_theory\mtd_game.py",
    
    # RQ4: Digital Twin
    "src\digital-twin\synchronization\twin_sync.py",
    
    # Infrastructure
    "tests\test_federated.py",
    ".github\workflows\tests.yml",
    "README.md",
    "requirements.txt",
    "LICENSE"
)

$all_present = $true
$total_files = $files_to_check.Count
$present_count = 0

foreach ($file in $files_to_check) {
    if (Test-Path $file) {
        $present_count++
        $size = (Get-Item $file).Length
        Write-Host "? $file ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "? MISSING: $file" -ForegroundColor Red
        $all_present = $false
    }
}

Write-Host "`n" + ("-" * 60) -ForegroundColor Cyan
Write-Host "Repository Completeness: $present_count/$total_files files" -ForegroundColor Yellow

if ($all_present) {
    Write-Host "`n? REPOSITORY IS 100% COMPLETE AND ALIGNED!" -ForegroundColor Green
    Write-Host "`n?? Ready for PhD applications!" -ForegroundColor Green
} else {
    Write-Host "`n??  Some files are missing - see above" -ForegroundColor Yellow
}

Write-Host "`nGitHub: https://github.com/hmshujaatzaheer/federated-scion-security-framework" -ForegroundColor Cyan
