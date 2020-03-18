import cv2

def set_camera(largura, altura, fps, brilho, cap):
    cap.set(3, largura)
    cap.set(4, altura)
    cap.set(cv2.CAP_PROP_GAIN, 100)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brilho)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    print("Setup WEBCAM feito")
