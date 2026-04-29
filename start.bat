@echo off
echo Starting AI Video Authenticity Detection System...

if not exist uploads mkdir uploads
if not exist models mkdir models

echo Starting FastAPI backend on port 8000...
start cmd /k "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000"

timeout /t 3 >nul

echo Starting Streamlit frontend on port 8501...
start cmd /k "cd frontend && streamlit run app.py"

echo.
echo System started!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:8501
pause
