import cv2

img = cv2.imread('/home/akhil/Downloads/cityscape1.png')
edges = cv2.Canny(img,100,200)

img_edge = cv2.resize(edges, (960, 540))
cv2.imshow("Canny- Edge Detector output", img_edge)
cv2.waitKey(0)