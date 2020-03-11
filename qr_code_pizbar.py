import cv2
import numpy as np
from pyzbar import pyzbar
import time
from setup_camera import set_camera

pi = 3.141592

def yaw_tratamento(yaw_ant, yaw, yaw_mod):
	if yaw-yaw_ant>0:
		horario=True
	else:
		horario=False
	if abs(yaw_ant-yaw)>pi/2*0.85:
		if horario==True:
			yaw_mod=yaw_mod-pi/2
		else:
			yaw_mod=yaw_mod+pi/2
	return yaw_mod


fps = 20.0
largura = 640
altura = 480
brilho = 130

cap = cv2.VideoCapture(0)
set_camera(largura, altura, fps, brilho, cap)

yaw_mod=0
yaw=0

while (cap.isOpened()):
	ret, img = cap.read()
	if ret == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#_,img = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
		qrs = pyzbar.decode(img)
		for qr in qrs:
			(a, b, c, d) = qr.polygon
			cv2.line(img, a, b, 0, 3)
			cv2.line(img, b, c, 0, 3)
			cv2.line(img, c, d, 0, 3)
			cv2.line(img, d, a, 0, 3)
			yaw_ant = yaw
			yaw = np.arctan2([b.y-a.y],[b.x-a.x])
			yaw_mod = yaw_tratamento(yaw_ant, yaw, yaw_mod)
			yaw_final = yaw+yaw_mod
			centro=[(a.x+b.x+c.x+d.x)/4, (a.y+b.y+c.y+d.y)/4]
			print("centro = %.2f %.2f guinada = %.2f" % (centro[0], centro[1], (yaw_final/pi*180)))
			
		cv2.imshow("adquiridos",img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	else:
		break
cap.release()
cv2.destroyAllWindows()
