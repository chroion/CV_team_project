import cv2
import numpy as np
import matplotlib.pyplot as plt

# 카메라 매트릭스 및 왜곡 계수 불러오기
with np.load('camera_params.npz') as file:
    K = file['mtx']
    dist = file['dist']

# ORB 초기화
orb = cv2.ORB_create()

# BFMatcher 객체 생성
bf = cv2.BFMatcher()

# 변수 초기화
saved_images = [] # 이미지 저장 여부
project_kp = False # 특징점 투영 여부

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    # 카메라 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 'c'를 눌러 현재 프레임 이미지 및 키포인트와 디스크립터 저장
    if cv2.waitKey(1) & 0xFF == ord('c') and len(saved_images) == 0:
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        saved_images = frame
        saved_keypoints = keypoints
        saved_descriptors = descriptors
        print("Image saved.")

    # 'm'을 눌러 현재 프레임과 저장된 이미지를 매칭하여 계속 표시
    if cv2.waitKey(1) & 0xFF == ord('m') and len(saved_images) > 0:
        project_kp = True
        
    # 특징점 투영
    if project_kp:
        # 현재 이미지의 키포인트와 디스크립터 저장
        current_keypoints, current_descriptors = orb.detectAndCompute(frame, None)
        
        # 현재 이미지와 저장된 이미지간의 디스크립터 매칭
        matches = bf.knnMatch(saved_descriptors, current_descriptors, k=2)
        
        # Lowe's ratio test 적용
        matches_good = [m1 for m1, m2 in matches if m1.distance < 0.70*m2.distance]
        sorted_matches = sorted(matches_good, key=lambda x: x.distance)

        # 매칭 결과로부터 좌표 추출
        query_idx = [match.queryIdx for match in matches_good]
        train_idx = [match.trainIdx for match in matches_good]

        p1 = np.float32([saved_keypoints[ind].pt for ind in query_idx])
        p2 = np.float32([current_keypoints[ind].pt for ind in train_idx])
        
        # Fundamental Matrix 계산 및 [R|t] 추정
        F, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_8POINT)
        E = K.T @ F @ K
        _, R, t, _ = cv2.recoverPose(E, p1, p2, K)

        # Triangulation을 통한 3D 좌표 계산
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = K @ np.hstack((R, t))
        tri_coord = cv2.triangulatePoints(P0, P1, p1.T, p2.T)
        tri_coord /= tri_coord[3]
        
        # 3D 포인트를 현재 프레임에 투영
        for i in range(tri_coord.shape[1]):
            # 3D 포인트
            point_3d = tri_coord[:3, i]
            
            # homogeneous 좌표로 변환
            point_3d_homogeneous = np.append(point_3d, 1)
            
            # 2D 투영
            point_2d_homogeneous = np.dot(P1, point_3d_homogeneous)
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            
            # 투영된 포인트를 이미지에 그리기
            x, y = int(point_2d[0]), int(point_2d[1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Camera Feed', frame)

    # 'q'을 눌러 카메라 이미지 표시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
