import os
import subprocess
import time
import requests
import streamlit as st
import pandas as pd

# ✅ MUST be the first st.* call
st.set_page_config(
    page_title="AI Video Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ✅ AUTO-START BACKEND if not running
def ensure_backend_running():
    try:
        requests.get(f"{API_URL}/", timeout=2)
        return True
    except:
        st.info("🚀 Starting backend server... Please wait...")
        backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
        subprocess.Popen(
            ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=backend_path,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        for _ in range(15):
            time.sleep(1)
            try:
                requests.get(f"{API_URL}/", timeout=2)
                return True
            except:
                continue
        return False

# Check backend at startup
if not ensure_backend_running():
    st.error("❌ Failed to start backend automatically.")
    st.stop()

st.success("✅ Backend connected!")

# ... rest of your code ...

API_URL = "http://localhost:8000"

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
MAX_FILE_SIZE_MB = 500


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f1f1f;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fake-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .real-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/", timeout=5)   # ✅ Fixed: use / not /health
        return response.status_code == 200
    except:
        return False


def upload_video(file):
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(f"{API_URL}/analyze", files=files, timeout=30)
    return response.json()


def get_result(task_id):
    response = requests.get(f"{API_URL}/result/{task_id}", timeout=10)
    return response.json()


# Header
st.markdown('<div class="main-header">🔍 AI Video Authenticity Detector</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deepfake Detection using Computer Vision & Deep Learning</div>', 
            unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system detects manipulated videos using:

    **Computer Vision Analysis:**
    - Eye Aspect Ratio (EAR) for blink detection
    - Face boundary artifact detection
    - Texture consistency analysis
    - Mouth movement analysis

    **Deep Learning:**
    - ResNet50/EfficientNet backbone
    - Transfer learning on deepfake datasets

    **Supported Formats:** MP4, AVI, MOV, MKV, WMV
    """)

    st.header("API Status")
    if check_api_health():
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Disconnected")
        st.info(f"Expected at: {API_URL}")

    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    if api_url != API_URL:
        os.environ["API_URL"] = api_url
        st.rerun()   # ✅ Fixed: st.rerun() not experimental_rerun

# Main content
st.subheader("📤 Upload Video")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="Upload a video to analyze for deepfake manipulation"
    )

with col2:
    st.info("""
    **Tips:**
    - Use clear, well-lit videos
    - Ensure face is visible
    - Max file size: 500MB
    - Analysis may take 30-120 seconds
    """)

if uploaded_file is not None:
    st.subheader("🎬 Video Preview")
    st.video(uploaded_file)

    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
    else:
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            if not check_api_health():
                st.error("Backend API is not running. Please start the backend server first.")
                st.info("Run: `cd backend && uvicorn main:app --reload`")
            else:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                with st.spinner("Uploading video..."):
                    try:
                        response = upload_video(uploaded_file)
                        task_id = response.get("task_id")

                        if not task_id:
                            st.error("Failed to start analysis task")
                        else:
                            status_placeholder.info(f"Task ID: `{task_id}`")

                            max_attempts = 60
                            for i in range(max_attempts):
                                time.sleep(2)

                                try:   # ✅ Added error handling
                                    result = get_result(task_id)
                                except Exception as e:
                                    status_placeholder.error(f"Connection lost: {e}")
                                    break

                                status = result.get("status")

                                if status == "completed":
                                    progress_placeholder.progress(100)
                                    status_placeholder.success("Analysis complete!")

                                    # Render results
                                    res = result.get("result", {})
                                    summary = res.get("summary", {})
                                    is_fake = res.get("is_fake", False)
                                    confidence = res.get("confidence", 0.0)
                                    message = res.get("message", "Unknown")
                                    frames = res.get("frames", [])

                                    st.divider()
                                    st.subheader("📊 Analysis Results")

                                    if is_fake:
                                        st.error(f"⚠️ {message} — Confidence: {confidence*100:.1f}% fake")
                                    else:
                                        st.success(f"✅ {message} — Confidence: {(1-confidence)*100:.1f}% real")

                                    c1, c2, c3 = st.columns(3)
                                    if is_fake:
                                        c1.metric("Fake Confidence", f"{confidence*100:.1f}%", delta="FAKE")
                                    else:
                                        c1.metric("Real Confidence", f"{(1-confidence)*100:.1f}%", delta="REAL")
                                    
                                    c2.metric("Frames Analyzed", summary.get("frames_analyzed", len(frames)))
                                    c3.metric("Video Duration", f"{summary.get('duration_seconds', 0):.1f}s")

                                    st.subheader("🔬 Detailed Metrics")
                                    mc1, mc2, mc3, mc4 = st.columns(4)
                                    with mc1:
                                        st.markdown("**👁️ Blink Anomaly**")
                                        blink = summary.get("blink_anomaly", 0)
                                        st.progress(min(max(float(blink), 0.0), 1.0))   # ✅ Clamped
                                        st.caption(f"Score: {blink:.2f}")
                                    with mc2:
                                        st.markdown("**🎨 Texture Anomaly**")
                                        texture = summary.get("texture_anomaly", 0)
                                        st.progress(min(max(float(texture), 0.0), 1.0))   # ✅ Clamped
                                        st.caption(f"Score: {texture:.2f}")
                                    with mc3:
                                        st.markdown("**🔲 Artifact Score**")
                                        artifact = summary.get("artifact_mean", 0)
                                        st.progress(min(max(float(artifact), 0.0), 1.0))   # ✅ Clamped
                                        st.caption(f"Score: {artifact:.2f}")
                                    with mc4:
                                        st.markdown("**👄 Mouth Anomaly**")
                                        mouth = summary.get("mouth_anomaly", 0)
                                        st.progress(min(max(float(mouth), 0.0), 1.0))   # ✅ Clamped
                                        st.caption(f"Score: {mouth:.2f}")

                                    if frames:
                                        st.subheader("📈 Frame-by-Frame Analysis")
                                        df = pd.DataFrame(frames)
                                        tab1, tab2, tab3, tab4 = st.tabs(["Eye Aspect Ratio", "Texture & Artifacts", "DL Score", "All Metrics"])   # ✅ Added DL tab
                                        with tab1:
                                            st.line_chart(df.set_index("frame_idx")[["ear", "mar"]], use_container_width=True)
                                            st.caption("EAR: Normal ~0.2-0.3. Constant values suggest no blinking (deepfake indicator).")
                                        with tab2:
                                            st.line_chart(df.set_index("frame_idx")[["texture_score", "artifact_score"]], use_container_width=True)
                                            st.caption("Lower texture scores may indicate over-smoothed deepfake faces.")
                                        with tab3:
                                            if "dl_score" in df.columns:
                                                st.line_chart(df.set_index("frame_idx")[["dl_score"]], use_container_width=True)
                                                st.caption("Deep learning confidence score (0=real, 1=fake)")
                                            else:
                                                st.info("DL score not available in this analysis")
                                        with tab4:
                                            st.dataframe(df, use_container_width=True)

                                    with st.expander("🔧 Technical Details"):
                                        st.json(summary)

                                    break

                                elif status == "failed":
                                    progress_placeholder.empty()
                                    status_placeholder.error(f"Analysis failed: {result.get('message')}")
                                    break
                                else:
                                    progress = min((i + 1) / max_attempts * 100, 95)
                                    progress_placeholder.progress(int(progress))
                                    status_placeholder.info(f"Analyzing... ({i*2}s elapsed)")
                            else:
                                status_placeholder.warning("Analysis timed out. Check results later.")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend API. Is it running?")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")