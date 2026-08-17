# run_ai_server.ps1 (TDTC-AI-CCTV)
# FastAPI 서버 실행 및 ngrok 터널링 자동 실행 스크립트

$Host.UI.RawUI.WindowTitle = "TDTC CCTV AI Server + ngrok"

# 1. 가상환경 활성화 (가상환경이 존재할 경우)
if (Test-Path "e:\AIVLE_10team\.venv") {
    Write-Host "[INFO] 가상환경(.venv) 활성화 중..." -ForegroundColor Cyan
    . "e:\AIVLE_10team\.venv\Scripts\Activate.ps1"
}

# 2. FastAPI 서버를 새로운 백그라운드 창에서 실행 (TDTC-AI-CCTV 폴더 기준)
Write-Host "[INFO] FastAPI AI 서버 시작 중 (Port: 8088)..." -ForegroundColor Green
$SERVER_CMD = "Set-Location 'e:\AIVLE_10team\TDTC-AI-CCTV'; if (Test-Path 'e:\AIVLE_10team\.venv') { . 'e:\AIVLE_10team\.venv\Scripts\Activate.ps1' }; python ai_server.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $SERVER_CMD

# 서버가 준비될 때까지 잠시 대기
Start-Sleep -Seconds 3

# 3. ngrok 터널링 실행
if (Test-Path "e:\AIVLE_10team\ngrok.exe") {
    & "e:\AIVLE_10team\ngrok.exe" http --domain=scenic-dander-nuttiness.ngrok-free.dev 8088
} else {
    Write-Host "[ERROR] e:\AIVLE_10team\ngrok.exe 파일을 찾을 수 없습니다. 수동으로 ngrok을 구동하세요." -ForegroundColor Red
}
