import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="Virtual Cloth Try-on",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1E3A8A;
        background: linear-gradient(90deg, #1E3A8A, #4F46E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }
    .instructions {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .instruction-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .instruction-element {
        color: #000000;
    }
    .upload-section {
        background-color: #EFF6FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #1E3A8A, #4F46E5);
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .webcam-section {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #F9FAFB;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #F3F4F6;
        border-radius: 10px;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .captured-image {
        margin-top: 20px;
        border: 2px solid #4F46E5;
        border-radius: 10px;
        padding: 10px;
    }
    .download-btn {
        background: linear-gradient(90deg, #10B981, #34D399);
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        border: none;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">Virtual Cloth Try-on</div>', unsafe_allow_html=True)

# Virtual Try-on Model Function
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

# Initialize session state
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'start_button_clicked' not in st.session_state:
    st.session_state.start_button_clicked = False
if 'stop_button_clicked' not in st.session_state:
    st.session_state.stop_button_clicked = False
if 'cloth_image_path' not in st.session_state:
    st.session_state.cloth_image_path = None
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'captured_image_bytes' not in st.session_state:
    st.session_state.captured_image_bytes = None

def start_webcam():
    st.session_state.webcam_running = True
    st.session_state.start_button_clicked = True
    st.session_state.stop_button_clicked = False

def stop_webcam():
    st.session_state.webcam_running = False
    st.session_state.stop_button_clicked = True

def capture_image():
    if st.session_state.webcam_running and st.session_state.latest_frame is not None:
        # Convert the image to bytes instead of saving to file
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(st.session_state.latest_frame, cv2.COLOR_RGB2BGR))
        if is_success:
            st.session_state.captured_image_bytes = buffer.tobytes()
            st.session_state.captured_image = st.session_state.latest_frame.copy()
            st.success("Image captured successfully!")
        else:
            st.error("Failed to capture image")

# Instructions Section
st.markdown("""
<div class="instructions">
    <div class="instruction-title">How It Works</div>
    <p class="instruction-element">Welcome to our Virtual Cloth Try-on application! Follow these simple steps to see how clothing will look on you in real-time:</p>
    <ol>
        <li><span class="instruction-element"><strong>Upload</strong> an image of the clothing item you want to try on</span></li>
        <li><span class="instruction-element"><strong>Adjust</strong> the color detection settings if needed (in sidebar)</span></li>
        <li><span class="instruction-element">Click the <strong>"Start Virtual Try-on"</strong> button</span></li>
        <li><span class="instruction-element">Allow camera access when prompted</span></li>
        <li><span class="instruction-element">Position yourself in front of the camera wearing a <strong>green t-shirt</strong> or stand in front of a <strong>green screen</strong></span></li>
        <li><span class="instruction-element">See yourself wearing the uploaded clothing in <strong>real-time!</strong></span></li>
        <li><span class="instruction-element">Click <strong>"Capture Image"</strong> to save your virtual try-on</span></li>
    </ol>
    <div class="info-box">
        <strong>Pro Tip:</strong> This application works best with a solid green t-shirt or green background for proper detection.
        Ensure good lighting conditions for best results.
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
uploaded_file = st.file_uploader("Choose a clothing image (JPEG, JPG, PNG)", type=["jpg", "jpeg", "png"])

cloth_image = None
if uploaded_file is not None:
    try:
        cloth_image = Image.open(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            cloth_image.save(temp_file.name)
            st.session_state.cloth_image_path = temp_file.name
        try:
            st.image(cloth_image, caption="Uploaded Clothing Item", use_container_width=True)
        except TypeError:
            st.image(cloth_image, caption="Uploaded Clothing Item")
        st.markdown('<div class="success-box">✅ Clothing image successfully uploaded! You can now start the virtual try-on.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.session_state.cloth_image_path = None
st.markdown('</div>', unsafe_allow_html=True)

# Start/Stop Buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.button("🚀 Start Virtual Try-on", key="start_button", on_click=start_webcam, disabled=st.session_state.cloth_image_path is None)
    if st.session_state.webcam_running:
        st.button("🛑 Stop Virtual Try-on", key="stop_button", on_click=stop_webcam)

# Webcam Feed
if st.session_state.webcam_running and st.session_state.cloth_image_path:
    st.markdown('<div class="webcam-section">', unsafe_allow_html=True)
    st.markdown('<div class="instruction-title">📸 Virtual Try-on</div>', unsafe_allow_html=True)
    st.markdown('<div class="success-box">Webcam is running! You can stop it with the button above.</div>', unsafe_allow_html=True)
    
    # Add Capture Image button
    st.button("📷 Capture Image", key="capture_button", on_click=capture_image)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Please check your camera settings and permissions.")
        st.session_state.webcam_running = False
    else:
        webcam_placeholder = st.empty()
        try:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from webcam")
                    break
                lower_green = np.array([lower_h, lower_s, lower_v])
                upper_green = np.array([upper_h, upper_s, upper_v])
                processed_frame = virtual_tryon(frame, st.session_state.cloth_image_path, lower_green, upper_green, min_area)
                st.session_state.latest_frame = processed_frame
                try:
                    webcam_placeholder.image(processed_frame, channels="RGB", use_container_width=True, caption="Virtual Try-on Preview")
                except TypeError:
                    webcam_placeholder.image(processed_frame, channels="RGB", caption="Virtual Try-on Preview")
                time.sleep(0.05)
                if not st.session_state.webcam_running:
                    break
        finally:
            cap.release()
            st.markdown('</div>', unsafe_allow_html=True)

# Display captured image and download button
if st.session_state.captured_image is not None and st.session_state.captured_image_bytes is not None:
    st.markdown('<div class="captured-image">', unsafe_allow_html=True)
    st.markdown('<div class="instruction-title">📷 Captured Image</div>', unsafe_allow_html=True)
    try:
        st.image(st.session_state.captured_image, use_container_width=True, caption="Your Virtual Try-on")
    except TypeError:
        st.image(st.session_state.captured_image, caption="Your Virtual Try-on")
    
    # Create download button using bytes
    btn = st.download_button(
        label="⬇️ Download Image",
        data=st.session_state.captured_image_bytes,
        file_name=f"virtual_tryon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        key="download_button"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.webcam_running and st.session_state.stop_button_clicked:
    st.markdown('<div class="info-box">Webcam stopped. You can start again by clicking the "Start Virtual Try-on" button.</div>', unsafe_allow_html=True)

# Tips Sidebar
st.sidebar.markdown("""
## 💡 Tips for Better Results

1. **Lighting**: Use good, even lighting  
2. **Green Color**: Use a vibrant green t-shirt or background  
3. **Camera Position**: Position yourself centrally in the frame  
4. **Adjustments**: Use the sliders to fine-tune color detection  

If the overlay isn't working:
- Tweak Hue, Saturation, and Value sliders
- Adjust Minimum Area
- Ensure green is clearly visible
""")

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2025 Virtual Cloth Try-on Project | Made with Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Clean up
if st.session_state.cloth_image_path and os.path.exists(st.session_state.cloth_image_path):
    try:
        os.unlink(st.session_state.cloth_image_path)
    except:
        pass
