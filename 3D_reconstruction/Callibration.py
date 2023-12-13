import numpy as np
import cv2

# 체커보드의 모서리 수 설정
corner_x = 6
corner_y = 4

# 3D 좌표를 저장할 배열
objpoints = []  # 실제 세계에서의 3D 포인트
imgpoints = []  # 이미지 평면에서의 2D 포인트

# 체커보드의 크기에 맞게 3D 좌표 생성
objp = np.zeros((corner_x * corner_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

# 카메라 캡처 시작
cap = cv2.VideoCapture(1)
save_data = False  # 데이터 저장 여부

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

    # 코너를 찾았다면
    if ret == True:
        # 캘리브레이션을 위한 포인트 수집
        if len(imgpoints) < 100:
            objpoints.append(objp)  # 3D 포인트 추가
            imgpoints.append(corners)  # 2D 포인트 추가
            print(len(imgpoints))

        # 충분한 포인트가 수집되면 캘리브레이션 수행
        elif len(imgpoints) == 100 and not save_data:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            np.savez('camera_params.npz', mtx=mtx, dist=dist)  # 결과 저장
            save_data = True  # 데이터 저장 완료
            print("Camera calibration data saved.")

    # 프레임을 화면에 표시
    cv2.imshow('frame', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()