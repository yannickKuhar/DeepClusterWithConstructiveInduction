import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from mat4py import loadmat

TAG = '[MNIST]'

def read_csv(file_name):
	indices = []

	f = open(file_name, "r")

	for line in csv.reader(f):
		indices.append( list(map(float, line)) )

	f.close()

	return indices[0]


def split_data(dataset, margin=0.7):
	n = math.ceil(len(dataset) * margin)

	learn = dataset[:n]
	test = dataset[n:]

	return (learn, test)


def one_to_two_dim(idx, img_dim):
	return idx // img_dim, idx % img_dim


def normalize_data(data):
    dmax = np.max(data)
    dmin = np.min(data)

    return (data - dmin) / (dmax - dmin)


def avg_image(data, construct, img_dim):
	init = [0.0] * (img_dim * img_dim)

	for d in data:
		for i in range(len(d)):
			init[construct[i]] += d[i]

	return np.array(normalize_data(init)).reshape((img_dim, img_dim)).T


def load_mnist():
	print(TAG, 'Loading data.')

	usps = loadmat('Data/mnist_fulldata.mat')
	X = np.array(usps['clusterdata'])
	Y = np.array(usps['clustertargets'])
	Y = np.array([int(np.where(t == 1)[0]) for t in Y])

	(learn_x, test_x) = split_data(X)
	(learn_y, test_y) = split_data(Y)

	print(TAG, 'Loading done.')

	return learn_x, test_x, learn_y, test_y


def split_by_class(data, classes):
	split = []

	for i in range(classes):
		split.append( list(np.where(data==i)[0]) )

	return split


def show_image(img):
	imgplot = plt.imshow(img, cmap='gray')
	plt.show()


def main(argv):
	construct_name = argv[1]
	labels_name = argv[2]

	classes = 10

	_, test_x, _, test_y = load_mnist()

	img_dim = int(math.sqrt(len(test_x[0])))
	print(TAG, 'Img dim: ', img_dim)

	construct = read_csv(construct_name)
	construct = list(map(int, construct))
	# print(TAG, construct)

	### USE LABELS FROM SVM!!! ###
	result_labels = read_csv(labels_name)
	result_labels = list(map(int, result_labels))
	# print(TAG, result_labels)

	print(TAG, 'Applying constrict start.')
	X = [x[construct] for x in test_x]
	Y = np.array(test_y)
	print(TAG, 'Applying constrict done.')
	print(TAG, 'Len X:', len(X), 'Dim X:', len(X[0]))

	splits = split_by_class(Y, classes)
	print( TAG, 'Splits Len:', len(splits), 'Len 0:', len(splits[0]), 'Len 1:', len(splits[1]) ) 
	# print(TAG, split)

	# results = [0] * 256

	# for c in construct:
		# results[c] = test_x[90][c]

	# results = np.array(results)
	# results = results.reshape((img_dim, img_dim)).T

	# show_image(results)

	for split in splits:
		data = [X[i] for i in split]
		img = avg_image(data, construct, img_dim)
		show_image(img)
		# print(TAG, 'Result img:', len(img), len(img[0]))
	
if __name__ == '__main__':
	main(sys.argv)