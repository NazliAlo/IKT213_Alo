import cv2


def print_image_information(image):
    print("Height:", image.shape[0])
    print("Width:", image.shape[1])
    print("Channels:", image.shape[2])
    print("Size:", image.size)
    print("Data type:", image.dtype)


def main():
    image = cv2.imread("../iris-1.jpg")
    print_image_information(image)

if __name__ == "__main__":
    main()