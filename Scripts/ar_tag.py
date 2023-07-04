import cv2
import numpy as np
import math
import traceback
import copy
import time
import argparse


def get_corners_from_contours(contours, corner_amount=4):
    coefficient = .05
    itr = 0
    while True:
        itr += 1
        epsilon = coefficient * cv2.arcLength(contours, True)

        poly_approx = cv2.approxPolyDP(contours, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == corner_amount:
            return hull
        else:
            if len(hull) > corner_amount:
                coefficient += .01
            else:
                coefficient -= .01
        if itr > 10:
            return []


def get_ar_tag_contours(contours, contour_hierarchy):
    paper_contours_ind = []
    ar_tag_contours = []
    for ind, contour in enumerate(contour_hierarchy[0]):
        if contour[3] == 0:
            paper_contours_ind.append(ind)

    if (len(paper_contours_ind) > 3):
        return None
    for ind in paper_contours_ind:
        ar_tag_contour_ind = contour_hierarchy[0][ind][2]
        ar_tag_contours.append(contours[ar_tag_contour_ind])

    return ar_tag_contours


def get_warped_tags(image, tag_to_corners_map, ar_tag_contours):
    dest_points = np.float32([[0, 0], [80, 0], [80, 80], [0, 80]])
    # warp image
    warped_tags = []

    # Creating an empty image out of the src img where warped tag pixels will be assigned
    dest_img = copy.deepcopy(image[:80, :80])
    dest_img[:, :] = 0

    for key in tag_to_corners_map:
        if len(tag_to_corners_map[key]) != 0:
            warp = get_warped_image(np.float32(tag_to_corners_map[key]), dest_points, image, dest_img, [], True)
            warped_tags.append(warp)
        else:
            warped_tags.append(None)

    return warped_tags


def get_warped_image(src_points, dest_points, src_img, dest_img, contour, is_dest_rect_plane):
    H = get_homography(src_points, dest_points)
    warped_img = warp_img_perspective(dest_points, H, src_img, dest_img, contour, is_dest_rect_plane)
    return warped_img


def get_ar_tag_id(tag_image, corner_list):
    _, binary = cv2.threshold(tag_image, 200, 255, cv2.THRESH_BINARY_INV)
    binary_reverse = cv2.bitwise_not(binary)
    # cv2.imshow(tag_str, binary_reverse)
    tag_corners_map = {}
    tag_corners_map["TL"] = tag_image[20:30, 20:30]  # tag_image[50:75, 50:75]
    tag_corners_map["TR"] = tag_image[20:30, 50:60]  # tag_image[50:75, 125:150]
    tag_corners_map["BR"] = tag_image[50:60, 50:60]  # tag_image[125:150, 125:150]
    tag_corners_map["BL"] = tag_image[50:60, 20:30]  # tag_image[125:150, 50:75]

    inner_corners_map = {}
    inner_corners_map["TL"] = tag_image[30:40, 30:40]  # tag_image[75:100, 75:100]
    inner_corners_map["TR"] = tag_image[30:40, 40:50]  # tag_image[75:100, 100:125]
    inner_corners_map["BR"] = tag_image[40:50, 40:50]  # tag_image[100:125, 100:125]
    inner_corners_map["BL"] = tag_image[40:50, 30:40]  # tag_image[100:125, 75:100]

    white_cell_corner = ''
    for cell_key in tag_corners_map:
        if is_cell_white(tag_corners_map[cell_key]):
            white_cell_corner = cell_key
            break
    if white_cell_corner == '':
        return None, corner_list

    print(white_cell_corner)
    id_number = [(is_cell_white(inner_corners_map[cell_key])) for cell_key in inner_corners_map]
    print('id_number', id_number)
    re_orient_action_map = {'TR': [3, 0, 1, 2], 'TL': [2, 3, 0, 1], 'BL': [1, 2, 3, 0], 'BR': [0, 1, 2, 3]}

    new_corners_list = []
    tag_id = 0
    for index, swap_ind in enumerate(re_orient_action_map[white_cell_corner]):
        new_corners_list.append(corner_list[swap_ind])
        tag_id = tag_id + id_number[swap_ind] * math.pow(2, (index))

    return tag_id, new_corners_list


def is_cell_white(cell):
    threshold = 200
    cell_to_gray = cv2.cvtColor((cell), cv2.COLOR_BGR2GRAY) if len(cell.shape) > 2 else cell
    return 1 if (np.mean(cell_to_gray) >= threshold) else 0


def warp_img_perspective(dest_points, H, src_img, dest_img, dest_contour, is_dest_rect_plane):
    # H - src to dest img
    H_inv = np.linalg.inv(H)
    H_inv = H_inv / H_inv[2][2]  # normalizing the inverse matrix

    dest_img_copy = copy.deepcopy(dest_img)
    src_img_dim = src_img.shape

    col_min, row_min = np.min(dest_points, axis=0)
    col_max, row_max = np.max(dest_points, axis=0)

    for y_ind in range(int(row_min), int(row_max)):
        for x_ind in range(int(col_min), int(col_max)):

            # we also can check whether the point is inside the contour or not before finding its mapping in the source image.
            # this can be done using (cv2.pointPolygonTest(dest_contour, (x_ind, y_ind), True) >= 0)
            # this will avoid mapping irrelavant points
            if is_dest_rect_plane or (
                    len(dest_contour) != 0 and cv2.pointPolygonTest(dest_contour, (x_ind, y_ind), True) >= 0):
                dest_pt = np.float32([x_ind, y_ind, 1]).T
                src_pt = np.dot(H_inv, dest_pt)
                src_pt = (src_pt / src_pt[2]).astype(int)
                if ((src_pt[1] in range(0, src_img_dim[0])) and (src_pt[0] in range(0, src_img_dim[1]))):
                    dest_img_copy[y_ind][x_ind] = src_img[src_pt[1]][src_pt[0]]

    return dest_img_copy


def get_homography(src_pts, dest_pts):
    A, b = [], []
    for i in range(4):
        src_x, src_y = src_pts[i]
        dest_x, dest_y = dest_pts[i]
        A_row_1 = np.asarray([src_x, src_y, 1., 0, 0, 0, -src_x * dest_x, -src_y * dest_x])
        A_row_2 = np.asarray([0, 0, 0, src_x, src_y, 1., -src_x * dest_y, -src_y * dest_y])
        A.append(A_row_1)
        A.append(A_row_2)
        b.append([dest_x])
        b.append([dest_y])

    A = np.asarray(A)
    b = np.asarray(b)

    h = np.linalg.lstsq(A, b)[0]
    H = []
    for elem in h:
        H.append(elem[0])
    H.append(1.)  # we assumed last element of H to be one (1)
    H = np.array(H).reshape((3, 3))
    return H

#
# def project_cube(dest_points, contour, src_img):
#     K = np.array([[1346.100595, 0, 932.1633975],
#                   [0, 1355.933136, 654.8986796],
#                   [0, 0, 1]]).T
#
#     src_points = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
#
#     H = get_homography(src_points, dest_points)
#     R_mat, t_vec = get_rotation_and_translation_matrix(K, H)
#     axis_points = np.float32([[0, 0, 0],
#                               [200, 0, 0],
#                               [200, 200, 0],
#                               [0, 200, 0],
#                               [0, 0, -200],
#                               [200, 0, -200],
#                               [200, 200, -200],
#                               [0, 200, -200]])
#
#     proj_corner_points, jacobian = cv2.projectPoints(axis_points, R_mat, t_vec, K, np.zeros((1, 4)))
#     dest_img = draw_cube(src_img, contour, proj_corner_points)
#     return dest_img


def get_rotation_and_translation_matrix(K, H):
    K_inv = np.linalg.inv(K)
    lam = (np.linalg.norm(np.dot(K_inv, H[:, 0])) + np.linalg.norm(np.dot(K_inv, H[:, 1]))) / 2
    lam = 1 / lam

    B_tilde = np.dot(K_inv, H)
    B = lam * B_tilde

    r1 = lam * B[:, 0]
    r2 = lam * B[:, 1]
    r3 = np.cross(r1, r2) / lam
    t = np.array([lam * B[:, 2]]).T
    R = np.array([r1, r2, r3]).T
    M = np.hstack([R, t])

    # P = np.dot(K, M)
    return R, t

#
# def draw_cube(img, bottom_contour, corner_points):
#     corner_points = np.int32(corner_points).reshape(-1, 2)
#
#     # draw bottom_contour
#     img = cv2.drawContours(img, bottom_contour, -1, (0, 255, 0), 3)
#
#     # draw lines to join bottom and top corners
#     for i, j in zip(range(4), range(4, 8)):
#         img = cv2.line(img, tuple(corner_points[i]), tuple(corner_points[j]), (255), 2)
#
#     # draw top_contour
#     img = cv2.drawContours(img, [corner_points[4:]], -1, (0, 0, 255), 2)
#     return img



def process_video():
    video_path = '/home/akhil/PycharmProjects/pythonProject/Perception/Project1_AR_Tags/1tagvideo.mp4'
    cap = cv2.VideoCapture(video_path)
    lena_img = cv2.imread("/home/akhil/Downloads/testudo.png")

    count = 0
    while True:
        try:
            # print('count', count)
            # count = count + 1
            # start = time.process_time()

            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break
            else:
                # img_copy_for_id = copy.deepcopy(image)
                # img_copy_for_lena = copy.deepcopy(image)
                # img_copy_for_cube = copy.deepcopy(image)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                # find the contours from the thresholded image
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                ar_tag_contours = get_ar_tag_contours(contours, hierarchy)
                if ar_tag_contours is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    corners = [get_corners_from_contours(contour) for contour in ar_tag_contours]

                    tag_to_corners_map = {}
                    for corner_ind, corner in enumerate(corners):
                        corner_list = []
                        if len(corner) != 0:
                            for ind, coordinate in enumerate(corner):
                                corner_list.append([coordinate[0][0], coordinate[0][1]])
                        tag_to_corners_map[corner_ind] = corner_list

                    warped_tags = get_warped_tags(gray, tag_to_corners_map, ar_tag_contours)

                    tag_num = 1
                    for ind, img in enumerate(warped_tags):
                        if img is not None:
                            # tag_str = 'tag-' + str(tag_num)
                            # tag_num = tag_num + 1
                            tag_id, corner_list = get_ar_tag_id(img, tag_to_corners_map[ind])
                            if tag_id is not None:
                                print("Tag ID :", tag_id)
                                tag_to_corners_map[ind] = corner_list
                                cv2.putText(frame, 'ID:' + str(tag_id),
                                            (corner_list[0][0] + 50, corner_list[0][1] + 10), font, 0.5, (200, 0, 155),
                                            1, cv2.LINE_AA)

                    # id_display_img = copy.deepcopy(frame)
                    con = cv2.drawContours(frame, ar_tag_contours, -1, (0, 255, 0), 2)
                    cv2.imshow('Tag and ID detection', con)

            if cv2.waitKey(100) == 27:
                break

        except Exception:
            traceback.print_exc()
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_video()
