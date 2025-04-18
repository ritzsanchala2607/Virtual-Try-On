import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Set page configuration
st.set_page_config(
    page_title="Virtual Cloth Try-on.",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
    <style>
    /* Your existing CSS styles here */
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">Virtual Cloth Try-on</div>', unsafe_allow_html=True)

# Virtual Try-on Model Function (same as before)
def virtual_tryon(frame, cloth_image_path, lower_green, upper_green, min_area):
    frame = cv2.flip(frame, 1)
    overlay_img = cv2.imread(cloth_image_path)
    if overlay_img is None:
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_tshirt = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if y < frame.shape[0] // 2:
                cv2.drawContours(mask_tshirt, [cnt], -1, 255, -1)

    overlay_resized = cv2.resize(overlay_img, (frame.shape[1], frame.shape[0]))
    tshirt_region = cv2.bitwise_and(overlay_resized, overlay_resized, mask=mask_tshirt)
    rest = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_tshirt))
    result = cv2.add(rest, tshirt_region)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# WebRTC Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255])
        self.min_area = 3000
        self.cloth_image_path = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        if self.cloth_image_path:
            processed_img = virtual_tryon(
                img, 
                self.cloth_image_path,
                self.lower_green,
                self.upper_green,
                self.min_area
            )
            return av.VideoFrame.from_ndarray(processed_img, format="rgb24")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Initialize session state
if 'cloth_image_path' not in st.session_state:
    st.session_state.cloth_image_path = None
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'captured_image_bytes' not in st.session_state:
    st.session_state.captured_image_bytes = None

# Instructions Section
st.markdown("""
<div class="instructions">
    <div class="instruction-title">How It Works (WebRTC Version)</div>
    <p class="instruction-element">This version uses browser-based webcam access that works in live deployments:</p>
    <ol>
        <li><span class="instruction-element"><strong>Upload</strong> a clothing image</span></li>
        <li><span class="instruction-element"><strong>Adjust</strong> color detection settings if needed</span></li>
        <li><span class="instruction-element"><strong>Allow camera access</strong> when prompted by your browser</span></li>
        <li><span class="instruction-element">Position yourself with a <strong>green background</strong></span></li>
        <li><span class="instruction-element">See the virtual try-on in real-time</span></li>
    </ol>
    <div class="info-box">
        <strong>Note:</strong> Requires HTTPS connection and camera permissions.
        Works best in Chrome/Firefox on desktop.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.title("⚙️ Color Detection Settings")
col1, col2 = st.sidebar.columns(2)
with col1:
    lower_h = st.slider("Lower Hue", 0, 179, 35)
    lower_s = st.slider("Lower Saturation", 0, 255, 40)
    lower_v = st.slider("Lower Value", 0, 255, 40)
with col2:
    upper_h = st.slider("Upper Hue", 0, 179, 85)
    upper_s = st.slider("Upper Saturation", 0, 255, 255)
    upper_v = st.slider("Upper Value", 0, 255, 255)
min_area = st.sidebar.slider("Minimum Area", 1000, 10000, 3000)

# Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="instruction-title">👕 Upload Clothing Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose clothing image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        cloth_image = Image.open(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            cloth_image.save(temp_file.name)
            st.session_state.cloth_image_path = temp_file.name
        st.image(cloth_image, caption="Uploaded Clothing Item", use_container_width=True)
        st.success("Clothing image uploaded successfully!")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.session_state.cloth_image_path = None
st.markdown('</div>', unsafe_allow_html=True)

# WebRTC Streamer
if st.session_state.cloth_image_path:
    st.markdown('<div class="webcam-section">', unsafe_allow_html=True)
    st.markdown('<div class="instruction-title">📸 Live Virtual Try-on</div>', unsafe_allow_html=True)
    
    ctx = webrtc_streamer(
        key="virtual-try-on",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        async_processing=True,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
    )
    
    if ctx.video_processor:
        ctx.video_processor.lower_green = np.array([lower_h, lower_s, lower_v])
        ctx.video_processor.upper_green = np.array([upper_h, upper_s, upper_v])
        ctx.video_processor.min_area = min_area
        ctx.video_processor.cloth_image_path = st.session_state.cloth_image_path
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Please upload a clothing image first")

# Tips Sidebar
st.sidebar.markdown("""
## 💡 Tips for Best Results

1. **Use Chrome/Firefox** for best compatibility
2. **Good lighting** is essential
3. **Solid green background** works best
4. **Reload page** if camera doesn't start
""")

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2025 Virtual Cloth Try-on Project | WebRTC Version</p>
    </div>
""", unsafe_allow_html=True)

# Clean up
if st.session_state.cloth_image_path and os.path.exists(st.session_state.cloth_image_path):
    try:
        os.unlink(st.session_state.cloth_image_path)
    except:
        pass
