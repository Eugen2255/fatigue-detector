import cv2
import mediapipe as mp
import os


''' Общий шаблон для работы с КОНКРЕТНЫМ изображением '''

# путь к изображению
path = os.path.abspath('')
img = cv2.imread(path)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow('image', img)
cv2.waitKey(0)




''' Общий шаблон для работы с камерой '''

cap = cv2.VideoCapture(0)

while True:

    suc, img = cap.read()
    if not suc: continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # передаем в метрики ВСЕ ключевые точки, а в отдельных функциях-метриках уже будем использовать только нужные

    '''
        1. тут будут функции возвращающие True/False
        2. потом будет скармливать ML общие флаги и на их основе будем определять степень усталости
        3. реализация интерфейса в конце
    '''

    cv2.imshow('image', img)

    key = cv2.waitKey(1) & 0xFF
    if key in ( ord('q'), 27): break


cv2.destroyAllWindows()
cap.release()