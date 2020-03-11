def dist_func(area,tamanho):
	dist = (18.35/area + 2.143)*tamanho
	return dist

def yaw_tratamento(yaw_ant, yaw, yaw_mod):
	pi = 3.141592
	if yaw-yaw_ant>0:
		horario=True
	else:
		horario=False
	if abs(yaw_ant-yaw)>pi/2*0.5:
		if horario==True:
			yaw_mod=yaw_mod-pi/2
		else:
			yaw_mod=yaw_mod+pi/2
	return yaw_mod

def area_poly(cantos):
	n = len(cantos)
	area = 0
	for i in range(n):
		j = (i+1) % n 
		area += cantos[i][0]*cantos[j][1]
		area -= cantos[j][0]*cantos[i][1]
	area = abs(area)/2
	return area
