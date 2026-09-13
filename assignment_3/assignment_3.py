import cv2
import numpy as np

image = cv2.imread("lambo.png")

# 1. Sobel edge detection
def sobel_edge_detection(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    sobel = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        1,
        1,
        ksize=1
    )

    sobel = cv2.convertScaleAbs(sobel)

    cv2.imwrite("sobel.png", sobel)

    return sobel


# 2. Canny edge detection
def canny_edge_detection(image, threshold_1, threshold_2):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, threshold_1, threshold_2)

    cv2.imwrite("canny.png", canny)

    return canny

# 3. Template matching
def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(
        image_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    threshold = 0.9

    locations = np.where(result >= threshold)

    template_height, template_width = template_gray.shape

    output = image.copy()

    for point in zip(*locations[::-1]):
        cv2.rectangle(
            output,
            point,
            (point[0] + template_width, point[1] + template_height),
            (0, 0, 255),
            2
        )

    cv2.imwrite("template_match.png", output)

    return output



# 4. Resizing using image pyramids
def resize(image, scale_factor: int, up_or_down: str):
    if scale_factor != 2:
        raise ValueError("scale_factor må være 2")

    if up_or_down == "up":
        resized = cv2.pyrUp(image)

    elif up_or_down == "down":
        resized = cv2.pyrDown(image)

    else:
        raise ValueError("up_or_down må være 'up' eller 'down'")


    filename = f"resize_{up_or_down}.png"

    cv2.imwrite(filename, resized)

    return resized

sobel_edge_detection(image)
canny_edge_detection(image, 50, 50)

shapes = cv2.imread("shapes-1.png")
template = cv2.imread("shapes_template.jpg")

template_match(shapes, template)

resize(image, 2, "up")
resize(image, 2, "down")