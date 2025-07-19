import cv2 as cv
import os
import numpy as np

# Checker board size
CHESS_BOARD_DIM = (9, 6)

# The size of Square in the checker board.
SQUARE_SIZE = 6  # millimeters

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

calib_data_path = "../calib_data"
CHECK_DIR = os.path.isdir(calib_data_path)

if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')
else:
    print(f'"{calib_data_path}" Directory already Exists.')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
obj_3D[:, :2] = np.mgrid[0:CHESS_BOARD_DIM[0], 0:CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
obj_3D *= SQUARE_SIZE
print(obj_3D)

# Arrays to store object points and image points from all the images.
obj_points_3D = []  # 3d point in real world space
img_points_2D = []  # 2d points in image plane.

# The images directory path
image_dir_path = "images"

files = os.listdir(image_dir_path)
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)

    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
    if ret:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)

        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

cv.destroyAllWindows()

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
)
print("Calibrated")

# Save calibration data
print("Dumping the data into one file using numpy")
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)
print("-------------------------------------------")

# Load calibration data
print("Loading data stored using numpy savez function\n")
data = np.load(f"{calib_data_path}/MultiMatrix.npz")
camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]
print("Loaded calibration data successfully")
