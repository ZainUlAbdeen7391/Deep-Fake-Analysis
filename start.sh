#!/bin/bash
# Start both backend and frontend

echo "Starting AI Video Authenticity Detection System..."

# Create necessary directories
mkdir -p uploads models

# Start backend in background
echo "Starting FastAPI backend on port 8000..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

sleep 3

# Start frontend
echo "Starting Streamlit frontend on port 8501..."
cd frontend
streamlit run app.py &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ System started!"
echo "📡 Backend API: http://localhost:8000"
echo "🖥️  Frontend UI: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
