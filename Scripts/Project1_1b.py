import cv2
import numpy as np
from matplotlib import pyplot as plt

# Inputing the reference marker
input = cv2.imread('ref_marker.png', cv2.IMREAD_UNCHANGED)

# Converting into grayscale and resizing it
imggray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
resize = cv2.resize(imggray, (8, 8))
imgthres = cv2.threshold(resize, 0, 1, cv2.THRESH_BINARY)
print("8*8 grid", imgthres[1])
cropped = imgthres[1][2:6, 2:6]
c = 0
for i in [cropped[0][0], cropped[0][3], cropped[3][0], cropped[3][3]]:
    if i == 1:
        break
    c += 1

# Decoding the 8*8 tag with values 0 and 1
val = c
#print(val)

# Function to detect ID and Orientation of the tag.
# Further, ChangeOrient function will rotate the tag as per the value.
def detect_id(image):
    orient = ''
    imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(imggray, 200, 255, cv2.THRESH_BINARY)
    img_unpadded = cv2.resize(img_binary, (8,8))
    img_unpadded = img_unpadded[2:6, 2:6]
    Test_binaryPoints = np.array([[1, 1], [2, 2], [1, 2], [2, 2]])
    Test_orientPoints = np.array([[3, 3], [0, 3], [0, 0], [3, 0]])
    white = 255
    binarylist = []

    for i in range(0, 4):
        x = Test_binaryPoints[i][0]
        y = Test_binaryPoints[i][1]
        if (img_unpadded[x][y]) == white:
            binarylist.append('1')
        else:
            binarylist.append('0')

    # If orient = 3, Rotate 90 Clockwise
    if img_unpadded[Test_orientPoints[0][0], Test_orientPoints[0][1]] == white:
        orient = 3

    #  If orient = 2, Rotate 180
    elif img_unpadded[Test_orientPoints[1][0], Test_orientPoints[1][1]] == white:
        orient = 2

    # If orient = 1, Rotate 90 counter clockwise
    elif img_unpadded[Test_orientPoints[2][0], Test_orientPoints[2][1]] == white:
        orient = 1

    elif img_unpadded[Test_orientPoints[3][0], Test_orientPoints[3][1]] == white:
        orient = 0
    returnstring = str(binarylist)
    plt.subplot(133), plt.imshow(img_unpadded, cmap="gray", vmin=0, vmax=1), plt.title("4x4 Grid")
    return returnstring, orient

print("Tagid and orientation", detect_id(input))

# Plotting various graph of tag
plt.subplot(131), plt.imshow(input, cmap="gray", vmin=0,vmax=1), plt.title("Refr_Marker")
plt.subplot(132), plt.imshow(resize, cmap="gray", vmin=0,vmax=1), plt.title("resized Image")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()