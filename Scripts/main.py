# Homework1 Question No.2 -- akhilv@umd.edu

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft, fftshift, ifftshift

vid_1 = cv2.VideoCapture('/home/akhil/PycharmProjects/pythonProject/Perception/Project1_AR_Tags/1tagvideo.mp4')
if vid_1.isOpened() == False:
    print('Error: Cannot Open Video file (ball_video1.mp4)')
width = int(vid_1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_1.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Circular HPF mask, center circle is 0, remaining all ones
def high_pass():
    highpass = np.ones((height, width, 2))
    radius = 80
    h = int(1080/2)
    k = int(1920/2)
    for i in range(0, width):
        for j in range(0, height):
            if (np.square(i - h) + np.square(j - k) < radius**2):
                highpass[i][j] = 0
    return highpass

# def create_circular_mask(h, w, center=None, radius=None):
#
#     if center is None: # use the middle of the image
#         center = (int(w/2), int(h/2))
#     if radius is None: # use the smallest distance between the center and image walls
#         radius = min(center[0], center[1], w-center[0], h-center[1])
#
#     Y, X = np.ogrid[:h, :w]
#     dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
#
#     mask = dist_from_center <= radius
#     return mask


def main():
    highpass = high_pass()

    # Circular HPF mask, center circle is 0, remaining all ones
    # rows, cols = 1080, 1920
    # crow, ccol = int(rows / 2), int(cols / 2)
    #
    # mask = np.ones((rows, cols, 2), np.uint8)
    # r = 80
    # center = [crow, ccol]
    # x, y = np.ogrid[:rows, :cols]
    # mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    # mask[mask_area] = 0

    while vid_1.isOpened():
        ret, frame = vid_1.read()
        frame_detect_copy = frame.copy()
        if not ret:
            break
        grey_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        forward_fft = cv2.dft(np.float32(grey_channel), flags = cv2.DFT_COMPLEX_OUTPUT)

        shifted_fft = fftshift(forward_fft)

        high_passed = shifted_fft * highpass

        shifted_i_fft = ifftshift(high_passed)
        final = cv2.idft(shifted_i_fft)
        final = cv2.magnitude(final[:, :, 0], final[:, :, 1])

        filltered_image = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        contours, hierarchy = cv2.findContours(filltered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # x, y, w, h = cv2.boundingRect(contours[2])
        # cv2.rectangle(frame_detect_copy, (x, y), (x + w, y + h), (255, 0, 0), 5)

        cv2.imshow('test', frame_detect_copy)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
