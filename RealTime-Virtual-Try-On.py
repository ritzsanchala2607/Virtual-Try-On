import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Load the image you want to overlay on the T-shirt
overlay_img = cv2.imread('./Pics/Garment-5.jpg')  
# Define green range (tune if needed)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask the green T-shirt
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.medianBlur(mask, 5)

    # Find contours to detect T-shirt region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_tshirt = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            if y < frame.shape[0] // 2:
                cv2.drawContours(mask_tshirt, [cnt], -1, 255, -1)

    # Resize the overlay image to the frame size (or better: tshirt bbox)
    overlay_resized = cv2.resize(overlay_img, (frame.shape[1], frame.shape[0]))

    # Blend the overlay only on the T-shirt area
    tshirt_region = cv2.bitwise_and(overlay_resized, overlay_resized, mask=mask_tshirt)
    rest = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_tshirt))

    result = cv2.add(rest, tshirt_region)

    cv2.imshow("T-Shirt Image Overlay", result)
    cv2.imshow("Original", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
