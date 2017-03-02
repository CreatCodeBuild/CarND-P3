"""
This is for analysis only. Has nothing to do with training and testing or driving.
"""
import glob
import os

from cv2 import imread, waitKey, imshow, resize
import cv2
from bokeh.models import Legend

from pandas import read_json
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot


def color_space_check():
	"""
	BGR Color Space
	"""
	image = imread('data/IMG/center_2016_12_01_13_30_48_287.jpg')
	imshow('', image)
	waitKey(100000)
	cv2.imwrite("example.jpg", image)
	print(image.shape)
	new = image[60:150, :, :]
	imshow('', new)
	cv2.imwrite("example_cropped.jpg", new)
	waitKey(100000)
	print(new.shape)
	new = resize(new, (200, 66))
	imshow('', new)
	cv2.imwrite("example_resized.jpg", new)
	waitKey(100000)
	print(new.shape)
	
# color_space_check()

figures = []
for path in glob.glob('./models/*.json'):
	df = read_json(path)
	p = figure(plot_width=800, plot_height=600, title=os.path.basename(path))
	p.line(df.index, df['loss'], color="firebrick", line_width=2, legend="Training")
	p.line(df.index, df['val_loss'], color="navy",  line_width=2, legend="Validation")
	figures.append(p)

output_file("training_history.html")
show(gridplot([[i] for i in figures]))
