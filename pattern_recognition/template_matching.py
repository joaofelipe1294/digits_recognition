import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2

class TemplateMatching(object):

	def __init__(self, train_values, train_labels, test_values, test_labels):
		self.train_values = train_values
		self.train_labels = train_labels
		self.test_values = test_values
		self.test_labels = test_labels
		self.K = 0
		self.templates = []
		self.template_labels = []

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
		for index in range(0 , len(labels)):
			if labels[index] == self.test_labels[index]:
				corrects += 1
		print('Precision : ' + str(float(corrects) / float(len(self.test_labels))))

	def calc_confusion_matrix(self, labels):
		confusion_matrix = np.zeros((10,10), np.uint32)
		for index in range(0, len(self.test_labels)):
			confusion_matrix[self.test_labels[index], labels[index]] += 1
		print('Confusion Matrix : ')
		print(confusion_matrix)

	def apply_with_avrage_images(self):
		self.read_values()
		self.calc_mean_images()
		train = self.prepare_values(self.templates)
		test = self.prepare_values(self.test_values)
		neigh = KNeighborsClassifier(n_neighbors = self.K)
		print('Treinando KNN ...')
		neigh.fit(train, self.template_labels)
		print('KNN treinado')
		print('Classificando imagens ...')
		labels = neigh.predict(test)
		print('Classificacao concluida')
		self.calc_precision(labels)
		self.calc_confusion_matrix(labels)

	def calc_mean_images(self):
		if sys.version_info[0] == 2:
			templates_number = int(raw_input('Templates por classe : '))
		else:
			templates_number = int(input('Templates por classe : '))
		clusters = [[], [], [], [] , [] , [] , [] , [] , [] , []]
		for index in range(0, len(self.train_labels)):
			clusters[self.train_labels[index]].append(self.train_values[index])
		images_by_template = len(clusters[0]) / templates_number
		for label in range(0, 10):
			for template in range(0, templates_number):
				avrage_image = np.zeros(self.train_values[0].shape, np.uint64)
				for index in range(template * images_by_template , (template + 1) * images_by_template):
					avrage_image += clusters[label][index]
				avrage_image = avrage_image / images_by_template
				avrage_image = np.array(avrage_image, np.uint8)
				self.templates.append(avrage_image)
				self.template_labels.append(label)
