#!/usr/bin/env python
"""
@author: Lars Schilling

"""
import rospkg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random as rng
from copy import copy
import math
from decimal import Decimal




def transform_data(data):

	data=np.array(data)
	data[data==0] = 1
	data[data==127] = 2
	data[data==255] = 0

	return data
def entropy(labels):
	counts = np.bincount(labels)
	norm_counts = counts / len(labels)

	entropy = 0

	for prob in norm_counts:
		if prob > 0:
			entropy += prob * np.log2(prob)

	return -entropy


if __name__ == '__main__':
	rospack = rospkg.RosPack()
	img = Image.open(rospack.get_path('heron_exploration')+'/sample_maps/sample_map_8.png')
	img_work = transform_data(img)
	total = np.size(img_work)
	plt.imshow(img_work)
	plt.show()
	info_img = img_work
	img_work[img_work==1]=0
	plt.imshow(info_img)
	plt.show()

	###try frontier point detection with img_work
	#find all frontier points, can be defined as edge detection problem, cv2.Canny can be used

	# Detect edges using Canny
	edges = cv2.Canny(img_work,10,10)
	max_Edge = np.max(edges)
	print max_Edge
	#img_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
	total1 = np.size(edges)
	print total1
	plt.imshow(edges)
	plt.show()
	edges_transformed = transform_data(edges)
	edge_pnts = np.asarray(np.where(edges==[255])).T.tolist()
	print edge_pnts
	print max(edge_pnts)
	#plt.scatter(x,y)

	#calculate information gain, tip: set everything to zero expect unexplored area
	#you can use cv2.filter2D with your own kernel
	kernel = np.ones((15,15),np.float32)/25

	smooth = cv2.filter2D(img_work, -1, kernel)
	max = smooth.argmax()
	print max
	print smooth.shape
	frontier = np.unravel_index(max, smooth.shape)
	print frontier
	x = frontier[1]
	y = frontier[0]
	odom.x= odom.pose.pose.position.x
	odom.y= odom.pose.pose.position.y	
	goal.x = odom.x + (x -32)*resolution
	goal.y = odom.y - (y -32)*resolution
				
	h, w = np.shape(smooth)
	total_smooth = np.size(smooth)


	plt.imshow(smooth)
	plt.show()
	max_frontier = cv2.circle(smooth, (frontier[1],frontier[0]), radius = 0, color= (139,0,0), thickness=-1)
	plt.imshow(max_frontier)
	plt.show()


	#find the frontier point with the biggest information gain, this will be our goal point
