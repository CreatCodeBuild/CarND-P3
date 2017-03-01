"""
This is for analysis only. Has nothing to do with training and testing or driving.
"""


from cv2 import imread, waitKey, imshow, resize
import cv2
# from scipy.misc import imread, imresize

def color_space_check():
	"""
	BGR Color Space
	"""
	image = imread('data/IMG/center_2016_12_01_13_30_48_287.jpg')
	imshow('', image)
	waitKey(100000)
	cv2.imwrite("example.jpg", image)
	print(image.shape)
	new = image[60:160, :, :]
	imshow('', new)
	cv2.imwrite("example_cropped.jpg", new)
	waitKey(100000)
	print(new.shape)
	new = resize(new, (200, 66))
	imshow('', new)
	cv2.imwrite("example_resized.jpg", image)
	waitKey(100000)
	print(new.shape)
	
color_space_check()
