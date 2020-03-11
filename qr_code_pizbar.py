import cv2
import numpy as np
from pyzbar import pyzbar
import time

img = cv2.imread("distorcido.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img[420:526, 420:600]  = 255
_,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
img=cv2.transpose(img)
img=cv2.flip(img,flipCode=2)



cv2.imshow("resultado", img)
cv2.waitKey(0)	
cv2.destroyAllWindows()

t_i = time.perf_counter()
qrs = pyzbar.decode(img)
print(time.perf_counter()-t_i)

for qr in qrs:
	(a, b, c, d) = qr.polygon
	cv2.line(img, a, b, 0, 2)
	cv2.line(img, b, c, 0, 2)
	cv2.line(img, c, d, 0, 2)
	cv2.line(img, d, a, 0, 2)
	
cv2.imshow("adquiridos",img)
cv2.waitKey(0)	
cv2.destroyAllWindows()
