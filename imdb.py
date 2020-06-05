import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from keras.preprocessing import sequence
from keras.datasets import cifar100, fashion_mnist, imdb

TAG = '[IMDB]'
WORD_CODES = 5000
INDEX_FROM=3 

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


def load_imdb():
	print(TAG, 'Loading data.')

	np_load_old = np.load
    # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

	(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=WORD_CODES, index_from=INDEX_FROM)
	max_words = 500
	X_train = sequence.pad_sequences(X_train, maxlen=max_words, padding='post')
	X_test = sequence.pad_sequences(X_test, maxlen=max_words, padding='post')
	X = np.concatenate((X_train, X_test), axis=0)
	Y = np.concatenate((y_train, y_test), axis=0)

	np.load = np_load_old

	(learn_x, test_x) = split_data(X)
	(learn_y, test_y) = split_data(Y)

	print(TAG, 'Loading done.')

	return learn_x, test_x, learn_y, test_y


def split_by_class(data, classes):
	split = []

	for i in range(classes):
		split.append( list(np.where(data==i)[0]) )

	return split


def most_common_n_words(corpus, n_words):
	word_dict = {c:0 for c in range(WORD_CODES + INDEX_FROM)}

	for doc in corpus:
		for word in doc:
			word_dict[word] += 1

	most_common_words = sorted(word_dict, key=word_dict.get, reverse=True)

	return most_common_words[:n_words]


def decode_words(words):

	word_to_id = imdb.get_word_index()
	word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
	word_to_id["<PAD>"] = 0
	word_to_id["<START>"] = 1
	word_to_id["<UNK>"] = 2
	word_to_id["<UNUSED>"] = 3
	id_to_word = {value:key for key,value in word_to_id.items()}

	decoded = {id_to_word[id] for id in words}

	return decoded


def write_words(file_name, words):
	with open(file_name, 'w') as fp:
		for w in words:
			fp.write(w + '\n')


def show_word_cloud(words, cmap):

	cloud = WordCloud(background_color='white', width=800, height=400, colormap=cmap).generate(words)

	plt.imshow(cloud)
	plt.axis('off')
	plt.show()


def main(argv):
	construct_name = argv[1]
	labels_name = argv[2]

	classes = 2
	n_words = 1000

	_, test_x, _, test_y = load_imdb()

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

	# 0 is - and 1 is +
	results = []

	for split in splits:
		data = [X[i] for i in split]
		words = most_common_n_words(data, n_words)
		decoded = decode_words(words)
		# print(TAG, 'Decoded words:', decoded)
		results.append(decoded)

	inter = results[0].intersection(results[1])

	print(TAG, 'Inter: ', inter)
	
	negative = ' '.join(list(results[0].difference(inter)))
	positive = ' '.join(list(results[1].difference(inter)))

	# print(TAG, 'Positive: ', positive)
	# print(TAG, 'Positive: ', negative)

	show_word_cloud(positive, 'Greens') 
	show_word_cloud(negative, 'Reds')


if __name__ == '__main__':
    main(sys.argv)