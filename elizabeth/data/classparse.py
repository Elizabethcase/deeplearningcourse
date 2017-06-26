#original by rykov8
#https://github.com/rykov8/ssd_keras/blob/master/PASCAL_VOC/get_data_from_XML.py

import numpy as np
import os
from xml.etree import ElementTree

CLASSES = ["n02802426", "container","water","bird bath", "tire", "wheelbarrow", "bucket", "gutter","vegetation"]

class XML_preprocessor(object):

    def __init__(self, data_path): #changed to input list of classes
        self.path_prefix = data_path
        self.num_classes = len(CLASSES)
        self.data = dict()
        self._preprocess_XML()
        self.classes = CLASSES

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            if filename.lower().endswith('.xml'):
                tree = ElementTree.parse(self.path_prefix + filename)
                root = tree.getroot()
                bounding_boxes = []
                one_hot_classes = []
                size_tree = root.find('size')
                width = float(size_tree.find('width').text)
                print("width =",width)
                height = float(size_tree.find('height').text)
                print("height =",height)
                #if width == 0:
                    #print("Res error for file", filename)
                #if height == 0:
                    #print("Res error for file", filename)
                for object_tree in root.findall('object'):
                    for bounding_box in object_tree.iter('bndbox'):
                        xmin = float(bounding_box.find('xmin').text)/width
                        ymin = float(bounding_box.find('ymin').text)/height
                        xmax = float(bounding_box.find('xmax').text)/width
                        ymax = float(bounding_box.find('ymax').text)/height
                    bounding_box = [xmin,ymin,xmax,ymax]
                    bounding_boxes.append(bounding_box)
                    class_name = object_tree.find('name').text
                    one_hot_class = self._to_one_hot(class_name)
                    one_hot_classes.append(one_hot_class)
                image_name = root.find('filename').text
                bounding_boxes = np.asarray(bounding_boxes)
                one_hot_classes = np.asarray(one_hot_classes)
                image_data = np.hstack((bounding_boxes, one_hot_classes))
                self.data[image_name] = image_data

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name in CLASSES:
            i = CLASSES.index(name)
            one_hot_vector[i] = 1
        else:
            print(name)
        return one_hot_vector
    

## example on how to use it
# import pickle
# data = XML_preprocessor('VOC2007/Annotations/').data
# pickle.dump(data,open('VOC2007.p','wb'))

