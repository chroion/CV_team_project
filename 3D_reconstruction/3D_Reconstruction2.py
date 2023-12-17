#3D_Reconstruction2

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def triangulate_points(P0, P1, pts1, pts2):
    num_points = pts1.shape[1]
    homogeneous_points_3d = np.zeros((4, num_points))

    for i in range(num_points):
        # 각 카메라에서의 특징점 좌표
        x1, y1 = pts1[:, i]
        x2, y2 = pts2[:, i]

        # 선형 시스템 구성
        A = np.vstack([
            x1 * P0[2, :] - P0[0, :],
            y1 * P0[2, :] - P0[1, :],
            x2 * P1[2, :] - P1[0, :],
            y2 * P1[2, :] - P1[1, :]
        ])

        # SVD를 사용하여 선형 시스템 해결
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]

        # 결과 저장
        homogeneous_points_3d[:, i] = X / X[3]

    return homogeneous_points_3d

def estimate_camera_pose(image, camera_matrix, dist_coeffs):
    pattern_size = (6, 4)
    square_size = 40.0  # 체스보드 스퀘어의 크기 (mm)

    found, corners = cv.findChessboardCorners(image, pattern_size)
    print("Corners found:", found)

    if found:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(cv.cvtColor(image, cv.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria)

        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        success, rvecs, tvecs = cv.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

        if success:
            return True, rvecs, tvecs
        else:
            print("solvePnP failed or returned unexpected results.")
            return False, None, None
    else:
        return False, None, None

def triangulate_points(kp1, kp2, good_matches, pose1, pose2, camera_matrix):
    P1 = pose1
    P2 = pose2

    P1 = camera_matrix @ P1
    P2 = camera_matrix @ P2

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    points_4d = triangulate_points(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3, :].T

    return points_3d, P2

def detect_and_match_features(img1, img2):
    AKAZE = cv.AKAZE_create()

    kp1, des1 = AKAZE.detectAndCompute(img1, None)
    kp2, des2 = AKAZE.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()

    return kp1, kp2, good_matches

# 카메라 매트릭스 및 왜곡 계수 불러오기
with np.load('C:/Users/Hyemin/Desktop/camera_params.npz') as file:
    K = file['mtx']
    dist = file['dist']

# AKAZE 초기화
AKAZE = cv.AKAZE_create()

# 변수 초기화
saved_images = []  # 이미지 저장 여부
project_kp = False  # 특징점 투영 여부

# 카메라 캡처 시작
cap = cv.VideoCapture(1)

while True:
    # 카메라 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 'c'를 눌러 현재 프레임 이미지 및 키포인트와 디스크립터 저장
    if cv.waitKey(1) & 0xFF == ord('c') and len(saved_images) == 0:
        saved_images = frame.copy()
        # 카메라 캘리브레이션을 위한 체스보드 코너 찾기
        saved_ret, saved_rvecs, saved_tvecs = estimate_camera_pose(saved_images, K, dist)
        if saved_ret:
            saved_R, _ = cv.Rodrigues(saved_rvecs)
            saved_RT = np.hstack((saved_R, saved_tvecs))
            print("Image saved.")
        else:
            print("Chessboard corners not found in saved image.")

    # 'm'을 눌러 현재 프레임과 저장된 이미지를 매칭하여 계속 표시
    if cv.waitKey(1) & 0xFF == ord('m') and len(saved_images) > 0:
        project_kp = True

    # 특징점 투영
    if project_kp:
        current_ret, current_rvecs, current_tvecs = estimate_camera_pose(frame, K, dist)

        if current_ret:
            kp1, kp2, good_matches = detect_and_match_features(saved_images, frame)

            current_R, _ = cv.Rodrigues(current_rvecs)
            current_RT = np.hstack((current_R, current_tvecs))

            points_3d, P2 = triangulate_points(kp1, kp2, good_matches, saved_RT, current_RT, K)

            for point in points_3d:
                point_3d_homogeneous = np.append(np.array(point), 1)
                # 2D 투영
                point_2d_homogeneous = np.dot(P2, point_3d_homogeneous)
                point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
                x, y = point_2d  # 각 포인트의 x, y 좌표 추출

                # 업데이트된 좌표를 이미지에 그리기
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv.imshow('Camera Feed', frame)

    # 'q'을 눌러 카메라 이미지 표시 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
