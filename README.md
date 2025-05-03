# Virtual Try-On

**Bridging Traditional Tailoring with Modern Technology**

This project introduces a **Virtual Try-On System** designed to help tailors and customers visualize how different clothing designs, fabrics, and colors will appear before actual stitching. The goal is to reduce miscommunication and increase satisfaction.

---
## Features

### 1) Image-Based Try-On
- Upload an image and preview different T-shirts virtually.
  
### 2) Real-Time Try-On (Green Clothing Support)
- Built using **MediaPipe Pose** for live augmented reality preview.
- Currently supports **green clothing** for overlay due to segmentation technique.

### 3) Optimized Try-On for All Colors (In Progress)
- A model that works for all color clothes.

---

## Technology Stack

- **Frontend**: Streamlit, HTML, CSS, Unity (for AR)
- **Backend**: Flask
- **AR Tools**: 
  - **Lens Studio** 
  - **Unity 3D** 
- **Pose Detection**: MediaPipe Pose

---

## AR Filters Developed

1. **Single Cloth Try-On Filter**  
   Allows users to try on one piece of clothing (e.g., a T-shirt).

2. **Single Cloth Material Changing Filter**  
   Users can change fabric materials using in-frame buttons.

3. **Multiple Clothes and Material Switching Filter** *(In Progress)*  
   Enables switching between multiple outfits and their fabrics in a single AR session.
