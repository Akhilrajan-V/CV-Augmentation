import cv2
import numpy as np


def main():
    path = '/home/akhil/PycharmProjects/pythonProject/Perception/Project1_AR_Tags/1tagvideo.mp4'
    video = cv2.VideoCapture(path)
    if video.isOpened() == False:
        print('Error: Cannot Open Video file')

    img_impose = cv2.imread('testudo.png')
    video_encode = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = 20
    output_video_1 = cv2.VideoWriter(str('Sample_testudo') + ".avi", video_encode, frame_rate, (1920, 1080))
    output_video_2 = cv2.VideoWriter(str('Sample_cube') + ".avi", video_encode, frame_rate, (1920, 1080))

    # Camera parameters
    K = np.array([[1346.100595, 0, 932.1633975],
                  [0, 1355.933136, 654.8986796],
                  [0, 0, 1]]).transpose()
    start_frame = 0
    video.set(1, start_frame)
    count =0
    while video.isOpened():
        ret, frame = video.read()
        if np.shape(frame) == ():
            break
        [conts, cont] = contour_det(frame)
        cv2.drawContours(frame, conts, -1, (255, 0, 0), 4)
        for i, tag in enumerate(cont):
            H = gen_homography_mat(tag)
            H_inv = np.linalg.inv(H)
            img_square = warp_tags(H_inv, frame, 200, 200)
            img_gray = cv2.cvtColor(img_square, cv2.COLOR_BGR2GRAY)
            ret, img_square = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

            [tag_id, pose] = detect_id(img_square)
            if count == 0:
                print("Detected AR Tag ID: ", tag_id)
                count += 1

            img_n = img_impose
            img_rotate = get_orientation(img_n, pose)
            size = img_rotate.shape[0]
            H = gen_homography_mat(tag, size)
            height = frame.shape[0]
            width = frame.shape[1]

            frame1 = warp_tags(H, img_rotate, height, width)
            frame2 = paste_image(frame, conts[i], 1)
            superimposed_frame = cv2.bitwise_or(frame1, frame2)
            cv2.imshow("Testudo superimposed Output", superimposed_frame)
            output_video_1.write(superimposed_frame)

            H1 = gen_homography_mat(tag, 200)
            H_inv1 = np.linalg.inv(H1)
            P = gen_project_mat(K, H_inv1)
            new_corners = cube(tag, H1, P, 200)
            frame = create_cube(tag, new_corners, frame, 0)

            cv2.imshow("AR cube Output", frame)
            output_video_2.write(frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def contour_det(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    ret, thresh = cv2.threshold(gray, 190, 255, 0)
    conts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    w_conts = []
    for i, height in enumerate(hierarchy[0]):
        if height[2] == -1 or height[3] == -1:
            w_conts.append(i)
    cont = [c for i, c in enumerate(conts) if i not in w_conts]

    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:3]
    return_cnts = []

    for c in cont:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * .015, True)
        if len(approx) == 4:
            return_cnts.append(approx)

    corners = []
    for shape in return_cnts:
        points = []
        for p in shape:
            points.append([p[0][0], p[0][1]])
        corners.append(points)

    return return_cnts, corners


def gen_homography_mat(corners, dim=200):
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    xp = [0, dim, dim, 0]
    yp = [0, 0, dim, dim]
    n = 9
    m = 8
    A_mat = np.empty([m, n])
    value = 0
    for row in range(0, m):
        if (row % 2) == 0:
            A_mat[row, 0] = -x[value]
            A_mat[row, 1] = -y[value]
            A_mat[row, 2] = -1
            A_mat[row, 3] = 0
            A_mat[row, 4] = 0
            A_mat[row, 5] = 0
            A_mat[row, 6] = x[value] * xp[value]
            A_mat[row, 7] = y[value] * xp[value]
            A_mat[row, 8] = xp[value]

        else:
            A_mat[row, 0] = 0
            A_mat[row, 1] = 0
            A_mat[row, 2] = 0
            A_mat[row, 3] = -x[value]
            A_mat[row, 4] = -y[value]
            A_mat[row, 5] = -1
            A_mat[row, 6] = x[value] * yp[value]
            A_mat[row, 7] = y[value] * yp[value]
            A_mat[row, 8] = yp[value]
            value += 1

    U, S, V = np.linalg.svd(A_mat)
    x = V[-1]
    H = np.reshape(x, [3, 3])
    return H


def detect_id(image):
    pose = ''
    ret, img_binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    depad = img_binary[50:150, 50:150]
    pose_point = np.array([[37, 37], [62, 37], [37, 62], [62, 62]])
    white = 255
    black = 0
    binarylist = []
    for i in range(0, 4):
        x = pose_point[i][0]
        y = pose_point[i][1]
        if (depad[x, y]) == white:
            binarylist.append(1)
        else:
            binarylist.append(0)
    if depad[pose_point[0][0], pose_point[0][1]] == white:
        pose = 3
    elif depad[pose_point[1][0], pose_point[1][1]] == white:
        pose = 2
    elif depad[pose_point[2][0], pose_point[2][1]] == white:
        pose = 1
    elif depad[pose_point[3][0], pose_point[3][1]] == white:
        pose = 0
    tag_id = str(binarylist)
    return tag_id, pose


def warp_tags(H, src, height, width):
    pnt_id, xx_id = np.indices((height, width), dtype=np.float32)
    in_ind = np.array([xx_id.ravel(), pnt_id.ravel(), np.ones_like(xx_id).ravel()])

    map_ind = H.dot(in_ind)
    x_map, y_map = map_ind[:-1] / map_ind[-1]
    x_map = x_map.reshape(height, width).astype(np.float32)
    y_map = y_map.reshape(height, width).astype(np.float32)

    x_map[x_map >= src.shape[1]] = -1
    x_map[x_map < 0] = -1
    y_map[y_map >= src.shape[0]] = -1
    x_map[y_map < 0] = -1

    return_img = np.zeros((height, width, 3), dtype="uint8")
    for x_new in range(width):
        for y_new in range(height):
            x = int(x_map[y_new, x_new])
            y = int(y_map[y_new, x_new])

            if x == -1 or y == -1:
                pass
            else:
                return_img[y_new, x_new] = src[y, x]
    return return_img


def get_orientation(image, pose):
    if pose == 1:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif pose == 2:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_180)
    elif pose == 3:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        reoriented_img = image
    return reoriented_img


def gen_project_mat(K, H):
    r_1 = H[:, 0]
    r_2 = H[:, 1]

    K = np.transpose(K)

    inv_K = np.linalg.inv(K)
    h1 = np.dot(inv_K, r_1)
    h2 = np.dot(inv_K, r_2)
    ld = 1 / ((np.linalg.norm(h1) + np.linalg.norm(h2)) / 2)

    B_T = np.dot(inv_K, H)

    if np.linalg.det(B_T) > 0:
        B_mat = 1 * B_T
    else:
        B_mat = -1 * B_T

    b_1 = B_mat[:, 0]
    b_2 = B_mat[:, 1]
    b_3 = B_mat[:, 2]
    r1 = ld * b_1
    r2 = ld * b_2
    r3 = np.cross(r1, r2)
    tq = ld * b_3
    Proj_mat = np.dot(K, (np.stack((r1, r2, r3, tq), axis=1)))
    return Proj_mat


def cube(corners, H, Proj_mat, dim):
    new_corners = []
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    s1 = np.stack((np.array(x), np.array(y), np.ones(len(x))))
    s2 = np.dot(H, s1)
    s3 = s2 / s2[2]

    projw = np.stack((s3[0], s3[1], np.full(4, -dim), np.ones(4)), axis=0)

    projsc = np.dot(Proj_mat, projw)
    cor = projsc / (projsc[2])
    for i in range(4):
        new_corners.append([int(cor[0][i]), int(cor[1][i])])

    return new_corners


def create_cube(tagcorners, new_corners, frame, flag):
    thickness = 3
    if not flag:
        contours = create_contours(tagcorners, new_corners)
        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (255, 255, 255), thickness=-1)

    for i, point in enumerate(tagcorners):
        cv2.line(frame, tuple(point), tuple(new_corners[i]), (0, 0, 255), thickness)

    for i in range(4):
        if i == 3:
            cv2.line(frame, tuple(tagcorners[i]), tuple(tagcorners[0]), (0, 255, 0), thickness)
            cv2.line(frame, tuple(new_corners[i]), tuple(new_corners[0]), (0, 255, 0), thickness)
        else:
            cv2.line(frame, tuple(tagcorners[i]), tuple(tagcorners[i + 1]), (0, 255, 0), thickness)
            cv2.line(frame, tuple(new_corners[i]), tuple(new_corners[i + 1]), (0, 255, 0), thickness)

    return frame


def paste_image(img, contour, color):
    cv2.drawContours(img, [contour], -1, color, thickness=-1)
    return img


def create_contours(corners1, corners2):
    contours = []
    for i in range(len(corners1)):
        if i == 3:
            con1 = corners1[i]
            con2 = corners1[0]
            con3 = corners2[0]
            con4 = corners2[i]
        else:
            con1 = corners1[i]
            con2 = corners1[i + 1]
            con3 = corners2[i + 1]
            con4 = corners2[i]
        contours.append(np.array([con1, con2, con3, con4], dtype=np.int32))
    contours.append(np.array([corners1[0], corners1[1], corners1[2], corners1[3]], dtype=np.int32))
    contours.append(np.array([corners2[0], corners2[1], corners2[2], corners2[3]], dtype=np.int32))

    return contours


if __name__ == "__main__":
    main()
