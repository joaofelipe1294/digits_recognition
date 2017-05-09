import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class TemplateMatching(object):

	def __init__(self, train_values, train_labels, test_values, test_labels):
		self.train_values = train_values
		self.train_labels = train_labels
		self.test_values = test_values
		self.test_labels = test_labels
		self.K = 0


	def read_values(self):
		if sys.version_info[0] == 2:
			self.K = int(raw_input('K value : '))
		else:
			self.K = int(input('K value : '))

	def prepare_values(self, values):
		prepared_values = []
		for image in values:
		 	prepared_values.append(np.ravel(image))
		return prepared_values

	def apply(self):
		self.read_values()
		train = self.prepare_values(self.train_values)
		test = self.prepare_values(self.test_values)
		neigh = KNeighborsClassifier(n_neighbors = self.K)
		print('Treinando KNN ...')
		neigh.fit(train, self.train_labels)
		print('KNN treinado')
		print('Classificando imagens ...')
		labels = neigh.predict(test)
		print('Classificacao concluida')
		self.calc_precision(labels)
		self.calc_confusion_matrix(labels)

	def calc_precision(self, labels):
		corrects = 0
		for index in xrange(0 , len(labels)):
			if labels[index] == self.test_labels[index]:
				corrects += 1
		print('Precision : ' + str(float(corrects) / float(len(self.test_labels))))

	def calc_confusion_matrix(self, labels):
		confusion_matrix = np.zeros((10,10), np.uint32)
		for index in range(0, len(self.test_labels)):
			confusion_matrix[self.test_labels[index], labels[index]] += 1
		print('Confusion Matrix : ')
		print(confusion_matrix)