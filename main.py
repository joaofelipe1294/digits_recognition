from base.base import load
import cv2
import numpy as np
from pattern_recognition.template_matching import TemplateMatching
import sys


train_path = sys.argv[1]
test_path = sys.argv[2]
train_images, train_labels, train_names = load(train_path, train_path + '.txt')
test_images, test_labels, test_names = load(test_path,  test_path + '.txt')

TemplateMatching(train_images, train_labels, test_images, test_labels).apply_with_avrage_images()
#TemplateMatching(train_images, train_labels, test_images, test_labels).apply()