import cv2
import numpy as np
from pyzbar import pyzbar
import time
from setup_camera import set_camera
from func_aux import dist_func, yaw_tratamento, area_poly


qr_tamanho = 9.1 #tamanho em centimetros

#Setup da webcam
fps = 20.0
largura = 640
altura = 480
brilho = 130
cap = cv2.VideoCapture(0)
set_camera(largura, altura, fps, brilho, cap)

#inicializacao de parametros necessarios
pi = 3.141592
yaw_mod=0
yaw=0

while (cap.isOpened()):
	t_in = time.perf_counter()
	ret, img = cap.read()
	if ret == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		qrs = pyzbar.decode(img)
		for qr in qrs:
			(a, b, c, d) = qr.polygon
			cv2.line(img, a, b, 0, 3)
			cv2.line(img, b, c, 0, 3)
			cv2.line(img, c, d, 0, 3)
			cv2.line(img, d, a, 0, 3)
			yaw_ant = yaw
			yaw = np.arctan2([d.y-a.y],[d.x-a.x])
			yaw_mod = yaw_tratamento(yaw_ant, yaw, yaw_mod)
			yaw_final = yaw+yaw_mod
			centro=[(a.x+b.x+c.x+d.x)/4, (a.y+b.y+c.y+d.y)/4]
			area=area_poly([a,b,c,d])/largura/altura*100
			print("area = %.2f dist = %.2f centro = %.2f %.2f guinada = %.2f frequencia = %.2f" % (area,dist_func(area,qr_tamanho), centro[0], centro[1], (yaw_final/pi*180), 1/(time.perf_counter()-t_in)))			
		#cv2.imshow("adquiridos",img)
		#if cv2.waitKey(1) & 0xFF == ord('q'):
		#		break
	else:
		break

cap.release()
cv2.destroyAllWindows()
