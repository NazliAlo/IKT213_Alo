import cv2


def get_camera_information(camera):
    fps = camera.get(cv2.CAP_PROP_FPS)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)

    return fps, height, width


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera.")
        return

    fps, height, width = get_camera_information(camera)

    print("FPS:", fps)
    print("Height:", height)
    print("Width:", width)

    with open("camera_outputs.txt", "w") as file:
        file.write(f"fps: {fps}\n")
        file.write(f"height: {height}\n")
        file.write(f"width: {width}\n")

    camera.release()


if __name__ == "__main__":
    main()