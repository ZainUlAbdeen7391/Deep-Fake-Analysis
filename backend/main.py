
import os
import uuid
import shutil
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector import DeepfakeDetector, VideoResult

# Configuration
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="AI Video Authenticity Detection API",
    description="Deepfake detection system using computer vision and deep learning",
    version="1.0.0"
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory result storage (use Redis in production)
results_store = {}

# Initialize detector
detector = DeepfakeDetector()


class AnalysisResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    message: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def _get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)


def analyze_video_task(task_id: str, file_path: str):
    """Background task for video analysis"""
    try:
        result = detector.analyze_video(file_path, sample_rate=5)

        # Convert to serializable format
        frame_data = []
        for f in result.frame_results:
            frame_data.append({
                "frame_idx": f.frame_idx,
                "has_face": f.has_face,
                "ear": round(f.ear, 4),
                "mar": round(f.mar, 4),
                "face_confidence": round(f.face_confidence, 4),
                "texture_score": round(f.texture_score, 4),
                "artifact_score": round(f.artifact_score, 4)
            })

        results_store[task_id] = {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "is_fake": bool(result.is_fake),           # ✅ Add bool()
                "confidence": float(result.confidence),    # ✅ Add float()
                "message": result.message,
                "summary": result.summary,
                "frames": frame_data
            },
            "message": "Analysis completed successfully"
        }
    except Exception as e:
        results_store[task_id] = {
            "task_id": task_id,
            "status": "failed",
            "result": None,
            "message": str(e)
        }
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="running",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )




@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a video file for deepfake analysis.
    Returns immediately with task_id. Poll /result/{task_id} for results.
    """
    # Validate file extension
    if not _allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    task_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}{file_ext}")

    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()

    # Check file size
    file_size = _get_file_size(file_path)
    if file_size > MAX_FILE_SIZE:
        os.remove(file_path)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
        )

    # Initialize result status
    results_store[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "result": None,
        "message": "Video analysis in progress"
    }

    # Start background analysis
    background_tasks.add_task(analyze_video_task, task_id, file_path)

    return AnalysisResponse(
        task_id=task_id,
        status="processing",
        result=None,
        message="Video uploaded and analysis started"
    )


@app.post("/analyze/sync")
async def analyze_video_sync(file: UploadFile = File(...)):
    """
    Synchronous video analysis. Waits for completion.
    Use for small videos (< 30 seconds) or testing.
    """
    if not _allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    task_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}{file_ext}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()

    file_size = _get_file_size(file_path)
    if file_size > MAX_FILE_SIZE:
        os.remove(file_path)
        raise HTTPException(status_code=413, detail="File too large")

    # Analyze directly
    try:
        result = detector.analyze_video(file_path, sample_rate=5)

        frame_data = []
        for f in result.frame_results:
            frame_data.append({
                "frame_idx": f.frame_idx,
                "has_face": f.has_face,
                "ear": round(f.ear, 4),
                "mar": round(f.mar, 4),
                "face_confidence": round(f.face_confidence, 4),
                "texture_score": round(f.texture_score, 4),
                "artifact_score": round(f.artifact_score, 4)
            })

        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "is_fake": bool(result.is_fake),           # ✅ Add bool()
                "confidence": float(result.confidence),    # ✅ Add float()
                "message": result.message,
                "summary": result.summary,
                "frames": frame_data
            },
            "message": "Analysis completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Get analysis result by task ID"""
    if task_id not in results_store:
        raise HTTPException(status_code=404, detail="Task not found")

    return results_store[task_id]


@app.delete("/result/{task_id}")
async def delete_result(task_id: str):
    """Delete analysis result"""
    if task_id in results_store:
        del results_store[task_id]
        return {"message": "Result deleted"}
    raise HTTPException(status_code=404, detail="Task not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
