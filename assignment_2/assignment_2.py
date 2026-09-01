import cv2
import numpy as np

image = cv2.imread("iris-1.png")
print(image.shape)

# Padding
def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )

    cv2.imwrite("padding.png", padded_image)
    return padded_image

padding(image, 100)

# Crop
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("crop.png", cropped_image)
    return cropped_image

crop(image, 200, 670, 200, 470)


# Resize
def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite("resize.png", resized_image)
    return resized_image
resize(image, 200, 200)

# Copy
def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]

    cv2.imwrite("copy.png", emptyPictureArray)
    return emptyPictureArray

height, width, channels = image.shape
emptyPictureArray = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

copy(image, emptyPictureArray)

def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("grayscale.png", grayscale_image)

    return grayscale_image
grayscale(image)


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imwrite("hsv.png", hsv_image)

    return hsv_image
hsv(image)

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = (
                     int(image[y, x, c]) + hue
                 ) % 256

    cv2.imwrite("hue_shifted.png", emptyPictureArray)

    return emptyPictureArray

height, width, channels = image.shape

emptyPictureArray = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)
hue_shifted(image, emptyPictureArray, 50)

def smoothing(image):
    smoothed_image = cv2.blur(image, (15, 15))

    cv2.imwrite("smoothing.png", smoothed_image)

    return smoothed_image
smoothing(image)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("rotation_90.png", rotated_image)

    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite("rotation_180.png", rotated_image)

    else:
        print("Invalid rotation angle")
        return None

    return rotated_image

rotation(image, 90)
rotation(image, 180)