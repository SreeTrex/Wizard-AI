import cv2
import time

# Initialize video capture (0 for webcam)
cap = cv2.VideoCapture(0)

# Set mobile view dimensions (9:16 aspect ratio)
MOBILE_WIDTH = 360
MOBILE_HEIGHT = 640

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Failed to capture first frame. Exiting.")
    cap.release()
    exit()

# Resize to mobile dimensions
prev_frame = cv2.resize(prev_frame, (MOBILE_WIDTH, MOBILE_HEIGHT))
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (15, 15), 0)  # Adjusted blur

motion_count = 0
last_saved_time = time.time()
SAVE_INTERVAL = 2  # Minimum 2 seconds interval

print("Motion detection started. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize to mobile view
    frame = cv2.resize(frame, (MOBILE_WIDTH, MOBILE_HEIGHT))
    display_frame = frame.copy()  # Frame for display only

    # Processing frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    # Motion detection
    frame_diff = cv2.absdiff(prev_frame, gray)
    _, thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)  # More sensitive
    
    # Improved motion detection with dilation
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Adjusted for mobile view
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save frame with timestamp
    if motion_detected and (time.time() - last_saved_time >= SAVE_INTERVAL):
        motion_count += 1
        filename = f"frame_{motion_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Motion detected! Saved as {filename}")
        last_saved_time = time.time()

    # Display mobile view
    cv2.imshow("Flutter App Emulator", display_frame)
    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Motion detection stopped.")
cap.release()
cv2.destroyAllWindows()