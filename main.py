from base import load
import cv2
import numpy as np

images, labels, names = load('bases/digits/train', 'bases/digits/train.txt')
#print(labels)


base_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for label in labels:
	base_labels[label] += 1

avrage_image = np.zeros(images[0].shape, np.uint64)
count = 0
for index in xrange(0, len(images)):
	if labels[index] == 8:
		avrage_image += images[index]
		count += 1

avrage_image /= count
avrage_image = np.array(avrage_image, np.uint8)
print(base_labels)
cv2.imshow('mean_image', avrage_image)
cv2.waitKey(0)