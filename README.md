# AI Video Authenticity Detection System

Deepfake detection system using computer vision and deep learning.
Built with **FastAPI** backend and **Streamlit** frontend.

## Features

- **Face Detection**: MediaPipe face mesh (468 landmarks)
- **Blink Analysis**: Eye Aspect Ratio (EAR) for unnatural blinking patterns
- **Texture Analysis**: Local Binary Patterns variance for skin texture consistency
- **Artifact Detection**: Face boundary and smoothing artifact detection
- **Mouth Movement**: Lip sync inconsistency detection
- **Deep Learning**: ResNet50/EfficientNet architecture (optional, requires training)

## Project Structure

```
deepfake-detector/
├── backend/
│   ├── main.py        # FastAPI application
│   ├── detector.py    # Core detection engine
│   └── model.py       # DL model architecture
├── frontend/
│   └── app.py         # Streamlit UI
├── models/            # Pre-trained weights storage
├── uploads/           # Temporary upload storage
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone/navigate to project
cd deepfake-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the System

### 1. Start Backend (FastAPI)

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 2. Start Frontend (Streamlit)

In a new terminal:

```bash
cd frontend
streamlit run app.py
```

Access at: http://localhost:8501

### 3. Using the System

1. Open Streamlit UI in browser
2. Upload a video (MP4, AVI, MOV, MKV, WMV)
3. Click "Analyze Video"
4. View results with frame-by-frame metrics

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Service status |
| POST | `/analyze` | Async video analysis (returns task_id) |
| POST | `/analyze/sync` | Synchronous video analysis |
| GET | `/result/{task_id}` | Get analysis result |
| DELETE | `/result/{task_id}` | Delete result |

## Deep Learning Model Training

To train the deep learning component:

1. Prepare dataset with `real/` and `fake/` folders
2. Install TensorFlow: `pip install tensorflow`
3. Run training script:

```python
from backend.model import create_model_for_training

model = create_model_for_training(
    data_dir="./dataset",
    backbone="efficientnetb0",
    output_dir="./models"
)
```

4. Place weights in `models/` directory
5. Update `detector.py` to load weights

## Detection Methodology

### Heuristic Analysis (No training required)

The system works out-of-the-box using computer vision heuristics:

1. **Eye Aspect Ratio (EAR)**: 
   - Normal: EAR varies as eyes blink
   - Deepfake: Often constant EAR (no blinking) due to training data bias

2. **Texture Analysis**:
   - Normal: Natural skin pore texture
   - Deepfake: Over-smoothed skin, low Laplacian variance

3. **Face Artifacts**:
   - Normal: Clean face boundaries
   - Deepfake: Blending artifacts at face edges

4. **Mouth Movement**:
   - Normal: Natural lip variation during speech
   - Deepfake: Unnatural or static mouth regions

### Deep Learning (Requires training)

- ResNet50/EfficientNetB0 backbone
- Transfer learning from ImageNet
- Fine-tuned on FaceForensics++ or Celeb-DF dataset
- Frame-level classification with temporal aggregation

## Performance Notes

- Analysis samples every 5th frame for performance
- Typical processing: 2-3 seconds per video second
- Supports videos up to 500MB
- Best results with frontal faces and good lighting

## Future Improvements

- [ ] Real-time webcam detection
- [ ] Audio-visual consistency analysis
- [ ] Temporal coherence modeling (LSTM/Transformer)
- [ ] Mobile app deployment
- [ ] Browser extension

## License

MIT License
