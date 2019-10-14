# Sample solution for task 2 (SIFT rotational invariance)
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


# Rotate an image
#
# image: image to rotate
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, angle):
    # Get center point of image to rotate around
    height, width = image.shape[:2]
    center = (height // 2, width // 2)
    
    # Compute rotation matrix
    rot_matrix = cv2.getRotationMatrix2D((center[1], center[0]), angle, 1.0)
    h, w = image.shape[:2]

    return cv2.warpAffine(image, rot_matrix, (w, h))


if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    image = cv2.imread("Eiffel_Tower.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rotate_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = SiftDetector(params=params)

    # Store SIFT keypoints of original image in a Numpy array
    kp = sift.detector.detect(gray, None)
    kp1, des1 = sift.detector.detectAndCompute(gray, None)
    gray = cv2.drawKeypoints(gray, kp1, None)
    cv2.imshow("Original keypoints", gray)
    cv2.waitKey(0)

    # Degrees with which to rotate image
    angle = -45

    # Rotate image
    rotate_gray = rotate(rotate_gray, angle)

    # Compute SIFT features for rotated image
    kp2, des2 = sift.detector.detectAndCompute(rotate_gray, None)
    kp_gray = cv2.drawKeypoints(rotate_gray, kp2, None)

    cv2.imshow("Rotated keypoints", kp_gray)
    cv2.waitKey(0)
