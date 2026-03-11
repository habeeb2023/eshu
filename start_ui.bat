@echo off
echo =============================================
echo  eshu — Start UI (Standalone)
echo =============================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0
echo.
echo Make sure Ollama is running: ollama serve
echo Models needed: phi3, nomic-embed-text, llava
echo.
echo Starting Streamlit on http://localhost:8501 ...
streamlit run ui/app.py --server.port 8501
pause
