# Sample solution for task 1 (Display SIFT features)
import cv2

from compute_sift import SiftDetector


# Parameters for SIFT initializations such that we find only 25% of keypoints
# relative to default settings (contrast threshold to 0.1)
params = {
    'n_features': 0,
    'n_octave_layers': 3,
    'contrast_threshold': 0.175,
    'edge_threshold': 10,
    'sigma': 1.6
}


if __name__ == '__main__':
    # 1. Read image
    image = cv2.imread("Eiffel_Tower.jpg")

    # 2. Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Initialize SIFT detector using default parameters
    sift = SiftDetector()
    
    # 4. Detect SIFT features
    kp = sift.detector.detect(gray, None)

    # 5. Visualize detected features
    kp_gray = cv2.drawKeypoints(image, kp, gray)
    cv2.imshow("Keypoints image", kp_gray)
    cv2.waitKey(0)
    
    # Print number of SIFT features detected
    print("Number of SIFT features (default settings): ", len(kp))
    
    # Get a new SIFT detector with parameters that detect less keypoints
    new_sift = SiftDetector(params=params)
    
    # Detect SIFT features
    new_kp = new_sift.detector.detect(gray, None)

    # Visualize detected features
    new_kp_gray = cv2.drawKeypoints(image, new_kp, gray)
    cv2.imshow("New keypoints image", new_kp_gray)
    cv2.waitKey(0)

    # Print number of SIFT features detected
    print("Number of SIFT features (new settings):     ", len(new_kp))

