@echo off
title Dallas Permit Bot Launcher

echo Starting Ollama...
start "Ollama" cmd /k "set OLLAMA_HOST=0.0.0.0 && ollama serve"
timeout /t 3 /nobreak

echo Starting ngrok...
start "ngrok" cmd /k "ngrok http 5678"
timeout /t 5 /nobreak

echo Setting webhook URL and starting n8n...
start "n8n" cmd /k "set WEBHOOK_URL=https://judgingly-satchel-subject.ngrok-free.dev && npx n8n start --tunnel"

echo.
echo All services started!
echo ngrok dashboard: http://localhost:4040
pause
