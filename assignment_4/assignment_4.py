import cv2
import numpy as np


# Harris Corner Detection
def harris_corner_detection(reference_image):

    # Convert image to grayscale
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Detect corners
    corners = cv2.cornerHarris(
        gray,
        2,
        3,
        0.04
    )

    # Make the corners more visible
    corners = cv2.dilate(corners, None)

    # Set a threshold
    threshold = 0.01 * corners.max()

    # Make a copy so the original image is not changed
    output = reference_image.copy()

    # Mark corners in red
    output[corners > threshold] = [0, 0, 255]

    # Save result
    cv2.imwrite("harris.png", output)

    return output



# Feature-Based Image Alignment using SIFT


def image_alignment(
        image_to_align,
        reference_image,
        max_features,
        good_match_precent
):

    # Convert both images to grayscale
    image_gray = cv2.cvtColor(
        image_to_align,
        cv2.COLOR_BGR2GRAY
    )

    reference_gray = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY
    )

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    image_keypoints, image_descriptors = sift.detectAndCompute(
        image_gray,
        None
    )

    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_gray,
        None
    )

    print("Image keypoints:", len(image_keypoints))
    print("Reference keypoints:", len(reference_keypoints))


    # FLANN matcher
    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    # Find the two best matches for each descriptor
    matches = flann.knnMatch(
        image_descriptors,
        reference_descriptors,
        k=2
    )

    good_matches = []

    for match_pair in matches:

        if len(match_pair) == 2:

            m, n = match_pair

            if m.distance < good_match_precent * n.distance:
                good_matches.append(m)

    print("Good matches:", len(good_matches))


    # Draw matches
    matches_image = cv2.drawMatches(
        image_to_align,
        image_keypoints,
        reference_image,
        reference_keypoints,
        good_matches,
        None,
        flags=2
    )

    cv2.imwrite(
        "matches.png",
        matches_image
    )


    # Find corresponding points

    src_points = np.float32([
        image_keypoints[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        reference_keypoints[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # Need at least four points for homography
    if len(good_matches) < 4:
        print("Not enough matches to calculate homography.")
        return None


    homography, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )


    # Align image
    height, width = reference_image.shape[:2]

    aligned_image = cv2.warpPerspective(
        image_to_align,
        homography,
        (width, height)
    )

    # Save aligned image
    cv2.imwrite(
        "aligned.png",
        aligned_image
    )

    return aligned_image




# Load images
reference_image = cv2.imread(
    "reference_img.png"
)

image_to_align = cv2.imread(
    "align_this.jpg"
)


# Run Harris Corner Detection
harris_corner_detection(
    reference_image
)


# Run SIFT image alignment
image_alignment(
    image_to_align,
    reference_image,
    10,
    0.7
)