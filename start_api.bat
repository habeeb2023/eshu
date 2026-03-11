@echo off
echo =============================================
echo  eshu — Start API Server
echo =============================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0
echo Starting FastAPI on http://localhost:8000 ...
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
