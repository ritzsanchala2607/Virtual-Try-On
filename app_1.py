import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="Virtual Cloth Try-on",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more attractive
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
    </style>
    """, unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">Virtual Cloth Try-on</div>', unsafe_allow_html=True)

# Virtual Try-on Model Function
def virtual_tryon(frame, cloth_image_path, lower_green, upper_green, min_area):
    """
    Apply virtual try-on of cloth image to the frame
    Args:
        frame: webcam frame
        cloth_image_path: path to the clothing image
        lower_green: lower bound for green color
        upper_green: upper bound for green color
        min_area: minimum contour area
    Returns:
        result: frame with the cloth overlay
    """
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Load the overlay image
    overlay_img = cv2.imread(cloth_image_path)
    if overlay_img is None:
        return frame
        
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mask with user-defined color range
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.medianBlur(mask, 5)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_tshirt = np.zeros_like(mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if y < frame.shape[0] // 2:
                cv2.drawContours(mask_tshirt, [cnt], -1, 255, -1)
    
    # Resize overlay
    overlay_resized = cv2.resize(overlay_img, (frame.shape[1], frame.shape[0]))
    
    # Blend
    tshirt_region = cv2.bitwise_and(overlay_resized, overlay_resized, mask=mask_tshirt)
    rest = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_tshirt))
    result = cv2.add(rest, tshirt_region)
    
    # Convert back to RGB for display
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Initialize session states
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'start_button_clicked' not in st.session_state:
    st.session_state.start_button_clicked = False
if 'stop_button_clicked' not in st.session_state:
    st.session_state.stop_button_clicked = False
if 'cloth_image_path' not in st.session_state:
    st.session_state.cloth_image_path = None

# Callback for start button
def start_webcam():
    st.session_state.webcam_running = True
    st.session_state.start_button_clicked = True
    st.session_state.stop_button_clicked = False

# Callback for stop button
def stop_webcam():
    st.session_state.webcam_running = False
    st.session_state.stop_button_clicked = True

# Instructions Section
st.markdown("""
    <div class="instructions">
        <div class="instruction-title">How It Works</div>
        <p>Welcome to our Virtual Cloth Try-on application! Follow these simple steps to see how clothing will look on you in real-time:</p>
        <ol>
            <li><strong>Upload</strong> an image of the clothing item you want to try on</li>
            <li><strong>Adjust</strong> the color detection settings if needed (in sidebar)</li>
            <li>Click the <strong>"Start Virtual Try-on"</strong> button</li>
            <li>Allow camera access when prompted</li>
            <li>Position yourself in front of the camera wearing a <strong>green t-shirt</strong> or stand in front of a <strong>green screen</strong></li>
            <li>See yourself wearing the uploaded clothing in <strong>real-time!</strong></li>
        </ol>
        <div class="info-box">
            <strong>Pro Tip:</strong> This application works best with a solid green t-shirt or green background for proper detection.
            Ensure good lighting conditions for best results.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Color adjustment sliders (for fine-tuning the green detection)
st.sidebar.title("⚙️ Color Detection Settings")
st.sidebar.markdown("Adjust these settings if the green detection isn't working well:")

col1, col2 = st.sidebar.columns(2)
with col1:
    lower_h = st.slider("Lower Hue", 0, 179, 35, key="lower_h")
    lower_s = st.slider("Lower Saturation", 0, 255, 40, key="lower_s")
    lower_v = st.slider("Lower Value", 0, 255, 40, key="lower_v")

with col2:
    upper_h = st.slider("Upper Hue", 0, 179, 85, key="upper_h")
    upper_s = st.slider("Upper Saturation", 0, 255, 255, key="upper_s")
    upper_v = st.slider("Upper Value", 0, 255, 255, key="upper_v")

min_area = st.sidebar.slider("Minimum Area", 1000, 10000, 3000, key="min_area")

# Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="instruction-title">👕 Upload Clothing Image</div>', unsafe_allow_html=True)
st.markdown('Please upload a clear image of the clothing item against a simple background for best results.', unsafe_allow_html=True)

# Create upload functionality
uploaded_file = st.file_uploader("Choose a clothing image (JPEG, JPG, PNG)", 
                                type=["jpg", "jpeg", "png"],
                                key="file_uploader")

# Display uploaded image if available
cloth_image = None
if uploaded_file is not None:
    try:
        cloth_image = Image.open(uploaded_file)
        # Save to session state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            cloth_image.save(temp_file.name)
            st.session_state.cloth_image_path = temp_file.name
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(cloth_image, caption="Uploaded Clothing Item", use_column_width=True)
            
        st.markdown('<div class="success-box">✅ Clothing image successfully uploaded! You can now start the virtual try-on.</div>', 
                   unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.session_state.cloth_image_path = None

st.markdown('</div>', unsafe_allow_html=True)

# Control buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    start_button = st.button("🚀 Start Virtual Try-on", 
                           key="start_button", 
                           on_click=start_webcam, 
                           disabled=st.session_state.cloth_image_path is None)

if st.session_state.webcam_running:
    with col2:
        stop_button = st.button("🛑 Stop Virtual Try-on", 
                              key="stop_button", 
                              on_click=stop_webcam)

# Webcam section
if st.session_state.webcam_running and st.session_state.cloth_image_path:
    st.markdown('<div class="webcam-section">', unsafe_allow_html=True)
    st.markdown('<div class="instruction-title">📸 Virtual Try-on</div>', unsafe_allow_html=True)
    
    # Create a status indicator
    st.markdown('<div class="success-box">Webcam is running! You can stop it with the button above.</div>', 
               unsafe_allow_html=True)
    
    # Webcam capture using OpenCV
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Cannot open webcam. Please check your camera settings and permissions.")
        st.session_state.webcam_running = False
    else:
        # Create a placeholder for the webcam feed
        webcam_placeholder = st.empty()
        
        # Process frames until webcam_running becomes False
        try:
            while st.session_state.webcam_running:
                # Read frame from webcam
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture image from webcam")
                    break
                
                # Get the user's color detection settings
                lower_green = np.array([lower_h, lower_s, lower_v])
                upper_green = np.array([upper_h, upper_s, upper_v])
                
                try:
                    # Process the frame using the virtual try-on function
                    processed_frame = virtual_tryon(frame, st.session_state.cloth_image_path, 
                                                  lower_green, upper_green, min_area)
                    
                    # Display the processed frame
                    webcam_placeholder.image(processed_frame, channels="RGB", 
                                            use_column_width=True, 
                                            caption="Virtual Try-on Preview")
                    
                    # Small pause to reduce CPU usage
                    time.sleep(0.05)
                    
                except Exception as e:
                    st.error(f"Error processing frame: {e}")
                    break
                
                # Check if we should stop
                if not st.session_state.webcam_running:
                    break
            
        finally:
            # Release resources
            cap.release()
            st.markdown('</div>', unsafe_allow_html=True)

# If webcam was running but now stopped
if not st.session_state.webcam_running and st.session_state.stop_button_clicked:
    st.markdown('<div class="info-box">Webcam stopped. You can start again by clicking the "Start Virtual Try-on" button.</div>', 
               unsafe_allow_html=True)

# Add tips in sidebar
st.sidebar.markdown("""
## 💡 Tips for Better Results

1. **Lighting**: Use good, even lighting
2. **Green Color**: Use a vibrant green t-shirt or background
3. **Camera Position**: Position yourself centrally in the frame
4. **Adjustments**: Use the sliders to fine-tune color detection if needed

If the clothing overlay isn't appearing properly:
- Try adjusting the Hue, Saturation, and Value sliders
- Increase/decrease the Minimum Area threshold
- Make sure your green shirt/background is clearly visible
""")

# Add footer
st.markdown("""
    <div class="footer">
        <p>© 2025 Virtual Cloth Try-on Project | Made with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Clean up temporary files when done
if st.session_state.cloth_image_path and os.path.exists(st.session_state.cloth_image_path):
    try:
        os.unlink(st.session_state.cloth_image_path)
    except:
        pass