import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect_tag(frame):
    img_blur = cv2.medianBlur(frame, 5)
    ret, thresh = cv2.threshold(img_blur, 190, 255, 0)
    all_con, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    w_conts = []
    for i, h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:
            w_conts.append(i)
    contour = [c for i, c in enumerate(all_con) if i not in w_conts]

    contour = sorted(contour, key=cv2.contourArea, reverse=True)[:3]

    val_cnt = []
    for c in contour:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * .015, True)
        if len(approx) == 4:
            val_cnt.append(approx)

    corners = []
    for shape in val_cnt:
        points = []
        for p in shape:
            points.append([p[0][0], p[0][1]])
        corners.append(points)

    return val_cnt, corners


def gen_mask (rows, cols):
    c_r = int(rows / 2)
    c_c = int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 100
    center = [c_r, c_c]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1
    return mask


def main():
    path = '/home/akhil/PycharmProjects/pythonProject/Perception/Project1_AR_Tags/1tagvideo.mp4'
    vid = cv2.VideoCapture(path)
    if vid.isOpened() == False:
        print('Error: Cannot Open Video file')

    vid.set(1, 100)
    ret, frame = vid.read()
    original_copy = frame.copy()
    grey_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(grey_scaled, 150, 255, cv2.THRESH_BINARY)
    rows, cols = grey_scaled.shape

    mask = gen_mask(rows, cols)  # Generating Mask

    # DFT and Inverse DFT
    dft = cv2.dft(np.float32(bin_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    forward_shift = dft_shift * mask
    f_shift_mask_mag = 2000 * np.log(cv2.magnitude(forward_shift[:, :, 0], forward_shift[:, :, 1]))
    f_ishift = np.fft.ifftshift(forward_shift)
    final = cv2.idft(f_ishift)
    final = cv2.magnitude(final[:, :, 0], final[:, :, 1])
    final = np.array(final)

    mini = final.min()
    maxi = final.max()
    a = 255 / (maxi - mini)
    b = 255 - a * maxi
    new_img = (a * final + b).astype('uint8')

    [conts, cnts] = detect_tag(new_img)
    cv2.drawContours(frame, conts, -1, (0, 255, 0), 4)

    plt.figure(1)
    plt.title("Original Image")
    plt.imshow(original_copy, cmap='gray')

    plt.figure(2)
    plt.title("Magnitude Spectrum ")
    plt.imshow(magnitude_spectrum, cmap='gray')

    plt.figure(3)
    plt.title("FFT + Mask ")
    plt.imshow(f_shift_mask_mag, cmap='gray')

    plt.figure(4)
    plt.title("Detected Tag")
    plt.imshow(frame)

    plt.figure(5)
    plt.title("After FFT and Inverse FFT")
    plt.imshow(final)

    # cv2.imshow("After FFT and Inverse FFT", final)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
