param (
    [Parameter(Position=0)]
    [ValidateSet("dev", "prod")]
    [string]$Profile = "prod"
)

$Host.UI.RawUI.WindowTitle = "TDTC CCTV AI Server + Cloudflare Tunnel ($Profile Mode)"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  [TDTC CCTV AI] FastAPI Server + Cloudflare Tunnel Start " -ForegroundColor Cyan
Write-Host "  Profile: $Profile (개발: dev / 운영: prod)" -ForegroundColor Yellow
Write-Host "  Domain : https://tdtc-ai-cctv.uk" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. FastAPI 서버 백그라운드 창에서 실행 (Port 8088)
Write-Host "[1/2] FastAPI AI Server Starting (Port: 8088, Profile: $Profile)..." -ForegroundColor Green
$ROOT = $PSScriptRoot
$SERVER_CMD = "Set-Location '$ROOT'; `$env:APP_ENV='$Profile'; python ai_server.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $SERVER_CMD

# 서버가 포트(8088)를 완전히 바인딩할 때까지 5초 대기
Start-Sleep -Seconds 5

# 2. Cloudflare 영구 고정 터널 실행 (tdtc-ai-cctv.uk)
Write-Host "[2/2] Cloudflare Tunnel Connecting (https://tdtc-ai-cctv.uk)..." -ForegroundColor Cyan

# .env.{profile} 또는 .env에서 토큰 파싱
$TOKEN = $env:CLOUDFLARE_TUNNEL_TOKEN
$ENV_FILE = if (Test-Path "$ROOT\.env.$Profile") { "$ROOT\.env.$Profile" } else { "$ROOT\.env" }
if (-not $TOKEN -and (Test-Path $ENV_FILE)) {
    Get-Content $ENV_FILE | ForEach-Object {
        if ($_ -match "^\s*CLOUDFLARE_TUNNEL_TOKEN\s*=\s*(.+)$") {
            $TOKEN = $matches[1].Trim()
        }
    }
}

if (-not $TOKEN) {
    Write-Host "[Warning] CLOUDFLARE_TUNNEL_TOKEN이 .env에 설정되지 않았습니다." -ForegroundColor Red
}

$CLOUDFLARED_CMD = "cloudflared"
if (Test-Path "C:\Program Files (x86)\cloudflared\cloudflared.exe") {
    $CLOUDFLARED_CMD = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
}
if (Test-Path "C:\Program Files\cloudflared\cloudflared.exe") {
    $CLOUDFLARED_CMD = "C:\Program Files\cloudflared\cloudflared.exe"
}

if ($TOKEN) {
    & $CLOUDFLARED_CMD tunnel run --token $TOKEN
} else {
    Write-Host "[Error] Cloudflare Tunnel Token이 없어 터널을 구동할 수 없습니다." -ForegroundColor Red
}
