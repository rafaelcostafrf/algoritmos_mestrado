import cv2
import numpy as np
import sys
import time

img = cv2.imread("distorcido.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def caixa_detec(im, caixa):
	n = len(caixa)
	for j in range(n):
		cv2.line(im, tuple(bbox[j][0]), tuple(bbox[(j+1) % n][0]), (255,0,0), 3)
		cv2.imshow("resultado", im)

detec_qr = cv2.QRCodeDetector()

t_i = time.perf_counter()
for i in range(10):
	data,bbox = detec_qr.detect(img)
print(time.perf_counter()-t_i)

if data==True:
	print("Dados Decodificados: {}".format(data))
	caixa_detec(img, bbox)

else:
	print("Nao foi detectado QRCode")
	cv2.imshow("Resultado", img)

cv2.waitKey(0)	
cv2.destroyAllWindows()

