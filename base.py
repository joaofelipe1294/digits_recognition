# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import cv2

#############################################################################################
#                                          LOAD                                             # 
#############################################################################################
# funcao que carrega as imagens de uma base (pasta)											#
# @recives																					#
# 	- uma string com o caminho da pasta que contem as imagens 							    #
#    																						#
# @returns                                                                                  #
# 	- tres listas, a primeira contem as imagens carregadas, objetos np.array, a segunda     #
# 	  contem inteiros 1 ou 0 referentes a classe da imagem,  a terceira contem strings      #
#     com o nome dos arquivos referentes as imagens                                         #
#############################################################################################

def load(base_path, file_path):
	print("Carregando imagens de " + base_path + " ...")
	file = open(file_path, 'r')
	names_txt = []
	labels_txt = []
	for line in file:
		line = line.split(' ')
		names_txt.append(line[0])
		labels_txt.append(int(line[1]))
	file.close()
	images = []
	labels = []
	names = []
	paths = os.listdir(base_path)
	paths.sort()
	for path in paths:
		image_path = base_path + '/' + path
		label = labels_txt[names_txt.index(path)]
		images.append(cv2.imread(image_path, 0))
		labels.append(label)
		names.append(path)
	print("Imagens carregadas")
	return images, labels, names

