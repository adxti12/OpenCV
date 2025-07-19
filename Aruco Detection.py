import cv2 as cv
from cv2 import aruco
import numpy as np

calib_data_path = "/home/aditi/calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 2.6  # centimeters

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
#aruco.DICT_4X4_50 is used, which contains 50 different ArUco markers of size 4x4 bits.

param_markers = aruco.DetectorParameters()# paramerters like markersize (threshhold paramters)

cap = cv.VideoCapture(2)

while True:
    ret, frame = cap.read()
    #ret - boolean (True if frame is read)
    #frame- 2d numpy array (RGB)
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)# make frame grey
    marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )
            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv.Rodrigues(rVec[i])

            # Extract Euler angles from rotation matrix
            euler_angles = cv.RQDecomp3x3(rotation_matrix)[0]

            # Draw Euler angles
            cv.putText(
                frame,
                f"Roll: {round(float(euler_angles[0]), 2)}",
                (20, 40),
                cv.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"Pitch: {round(float(euler_angles[1]), 2)}",
                (20, 60),
                cv.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"Yaw: {round(float(euler_angles[2]), 2)}",
                (20, 80),
                cv.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
