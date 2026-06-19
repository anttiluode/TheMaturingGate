import cv2
import numpy as np

def run_webcam_mirror():
    print("==========================================================")
    print("THE WEBCAM MIRROR GATE")
    print("Left: Raw World | Center: Prediction | Right: 'Noise' (Surprise)")
    print("Press ESC to exit.")
    print("==========================================================")

    # Resolution. We downscale to 64x64 so the math runs fast in real-time CPU.
    # 64x64 = 4096 pixels. The Operator matrix F will be 4096 x 4096 (~16 million params)
    dim = 64
    N = dim * dim
    
    # Learning rate of the fast network. 
    # High enough to learn motion, low enough not to be instantly corrupted by static.
    lr = 0.15

    # The Fast Network (Forward Model). 
    # We initialize it as an Identity matrix (it assumes the next frame equals the current frame)
    F = np.eye(N, dtype=np.float32)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Grab the very first frame to initialize the state
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (dim, dim))
    x_prev = small.flatten().astype(np.float32) / 255.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Process the raw world
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (dim, dim))
        x_curr = small.flatten().astype(np.float32) / 255.0

        # 2. THE PREDICTION: The fast network guesses the current frame from the previous one
        x_pred = F @ x_prev

        # 3. THE NOISE (Surprise): What the network failed to predict
        surprise = x_curr - x_pred

        # 4. THE LEARNING: Update the fast network using the Normalized Least Mean Squares (Delta Rule)
        # It corrects its internal matrix based on the surprise it just experienced.
        norm_sq = np.dot(x_prev, x_prev) + 1e-6
        F += lr * np.outer(surprise, x_prev) / norm_sq

        # --- Display Prep ---
        # Reshape flat vectors back into 2D images
        img_raw = x_curr.reshape(dim, dim)
        img_pred = np.clip(x_pred.reshape(dim, dim), 0, 1)
        
        # Amplify the surprise by 5x so it's highly visible, take absolute value
        img_noise = np.clip(np.abs(surprise.reshape(dim, dim)) * 5.0, 0, 1)

        # Stitch them together horizontally
        stitched = np.hstack([img_raw, img_pred, img_noise])
        
        # Scale the 64x192 stitched image up to a readable size on your monitor (nearest neighbor to keep it crisp)
        display = cv2.resize(stitched, (dim * 3 * 6, dim * 6), interpolation=cv2.INTER_NEAREST)

        # Draw Labels
        cv2.putText(display, "Raw World", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0,), 1)
        cv2.putText(display, "Prediction", (dim*6 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0,), 1)
        cv2.putText(display, "The 'Noise' (Surprise)", (dim*12 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0,), 1)

        cv2.imshow("The Mirror Gate", display)

        # Shift time forward
        x_prev = x_curr

        # Press ESC to break
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_mirror()