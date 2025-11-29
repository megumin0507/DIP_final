import cv2
import numpy as np

def main():

    cam_index = 2
    pattern_size = (9, 6)
    square_size = 1.0
    num_samples_target = 30

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
    
    print("Press 'q' to quit.")
    print("Move the checkerboard around the view to get different poses.")

    samples_collected = 0
    last_had_pattern = False

    while samples_collected < num_samples_target:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_FAST_CHECK
        )

        if found:
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria
            )

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, pattern_size, corners_refined, found)

            if not last_had_pattern:
                objpoints.append(objp.copy())
                imgpoints.append(corners_refined)
                samples_collected += 1
                print(f"Captured sample {samples_collected}/{num_samples_target}")

                last_had_pattern = True
        else:
            last_had_pattern = False

        cv2.putText(
            frame,
            f"Samples: {samples_collected}/{num_samples_target}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Calibration Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- Calibration ---
    if len(objpoints) < 3:
        print("Not enough samples collected for a stable calibration.")
        return
    
    # Use image size from the last frame processed
    image_size = gray.shape[::-1]  # (width, height)

    print("Running camera calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    print("\nReprojection error (RMS):", ret)
    print("\nCamera Matrix (K):\n", camera_matrix)
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3, ...):\n", dist_coeffs)

    # Optionally save calibration results
    np.savez(
        "calibration_result.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        objpoints=objpoints,
        imgpoints=imgpoints,
        image_size=image_size
    )
    print("\nCalibration data saved to calibration_result.npz")

if __name__ == "__main__":
    main()