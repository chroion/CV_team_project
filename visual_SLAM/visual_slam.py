import cv2
import numpy as np
import matplotlib.pyplot as plt

# 카메라 매트릭스 및 왜곡 계수 불러오기
with np.load('./camera_params.npz') as file:
    K = file['mtx']
    dist = file['dist']

# ORB 초기화
orb = cv2.ORB_create()

# BFMatcher 객체 생성
bf = cv2.BFMatcher()

# 초기 설정
saved_images = [] # 저장된 이미지
project_kp = False # 특징점 투영 여부
tri_coord_dict = {}  # 각 특징점의 월드좌표를 저장하는 딕셔너리
alpha = 0.8     # 이동 평균 강도 파라미터
max_distance = 10.0  # 좌표 업데이트를 위한 최대 거리 임계값
prev_R = np.eye(3)  # 이전 회전 행렬 초기화 (단위 행렬)
prev_t = np.zeros((3, 1))  # 이전 이동 벡터 초기화
prev_2d_points = {}  # 이전에 투영된 2D 포인트를 저장하는 딕셔너리

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
        
        if len(p2) == 0:
            continue
        
        # Fundamental Matrix 계산 및 [R|t] 추정
        F, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_8POINT)
        
        # F의 형태가 (3, 3)이 아니면 계산을 스킵
        if F is None or F.shape != (3, 3):
            continue
        
        E = K.T @ F @ K
        _, R, t, _ = cv2.recoverPose(E, p1, p2, K)

        # R, t의 평균 계산
        R_avg = alpha * R + (1 - alpha) * prev_R
        t_avg = alpha * t + (1 - alpha) * prev_t

        # 이전 자세 업데이트
        prev_R = R
        prev_t = t

        # Triangulation을 통한 3D 좌표 계산
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = K @ np.hstack((R_avg, t_avg))
        tri_coord = cv2.triangulatePoints(P0, P1, p1.T, p2.T)
        tri_coord /= tri_coord[3]
        
        # 새로 계산된 특징점의 월드좌표를 기존 리스트와 비교하여 업데이트
        for match, coord in zip(matches_good, tri_coord.T):
            query_idx = match.queryIdx
            world_coord = tuple(coord[:3])  # 좌표를 튜플로 변환

            # 매칭된 특징점 좌표가 딕셔너리에 존재하는지 확인
            if query_idx in tri_coord_dict:
                existing_coord = tri_coord_dict[query_idx]
                distance = np.linalg.norm(np.array(existing_coord) - np.array(world_coord))

                # 거리가 임계값보다 작은 경우에만 이동 평균으로 좌표 업데이트
                if distance < max_distance:
                    new_coord = tuple(alpha * np.array(world_coord) + (1 - alpha) * np.array(existing_coord))
                    tri_coord_dict[query_idx] = new_coord
            else:
                # 새로운 좌표 추가
                tri_coord_dict[query_idx] = world_coord
        
        # 딕셔너리에 저장된 3D 포인트를 현재 프레임에 투영
        for key, point_3d in tri_coord_dict.items():
            # homogeneous 좌표로 변환
            point_3d_homogeneous = np.append(np.array(point_3d), 1)
            
            # 2D 투영
            point_2d_homogeneous = np.dot(P1, point_3d_homogeneous)
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            
            # 이동 평균을 사용하여 투영된 포인트 업데이트
            new_x, new_y = point_2d
            if key in prev_2d_points:
                prev_x, prev_y = prev_2d_points[key]
                avg_x = alpha * new_x + (1 - alpha) * prev_x
                avg_y = alpha * new_y + (1 - alpha) * prev_y
            else:
                avg_x, avg_y = new_x, new_y
            
            # 업데이트된 좌표를 이미지에 그리기
            if 0 <= avg_x < frame.shape[1] and 0 <= avg_y < frame.shape[0]:
                cv2.circle(frame, (int(avg_x), int(avg_y)), 5, (0, 255, 0), -1)

            # 이전 좌표 업데이트
            prev_2d_points[key] = (avg_x, avg_y)

    
    cv2.imshow('Camera Feed', frame)

    # 'q'을 눌러 카메라 이미지 표시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
