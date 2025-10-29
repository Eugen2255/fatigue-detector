import cv2

from pose_utils import found_kp 


# пути к модели и изображению (добавьте ваши пути для тестов)
model_path = ".../pose_landmarker_lite.task"
image_path = ".../.jpg"


img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")


poses_kps = found_kp(img, model_path)

for pose in poses_kps:
    for x,y in pose:
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)


cv2.imshow("result image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
