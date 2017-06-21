#original by rykov8
#https://github.com/rykov8/ssd_keras/blob/master/PASCAL_VOC/get_data_from_XML.py

import numpy as np
import os
from xml.etree import ElementTree

class XML_preprocessor(object):

    def __init__(self, data_path, CLASSNUM=29): #added CLASSNUM
        self.path_prefix = data_path
        self.num_classes = CLASSNUM #changed from 20
        self.data = dict()
        self._preprocess_XML()

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
                height = float(size_tree.find('height').text)
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
        if name == 'aeroplane':
            one_hot_vector[0] = 1
        elif name == 'bicycle':
            one_hot_vector[1] = 1
        elif name == 'bird':
            one_hot_vector[2] = 1
        elif name == 'boat':
            one_hot_vector[3] = 1
        elif name == 'bottle':
            one_hot_vector[4] = 1
        elif name == 'bus':
            one_hot_vector[5] = 1
        elif name == 'car':
            one_hot_vector[6] = 1
        elif name == 'cat':
            one_hot_vector[7] = 1
        elif name == 'chair':
            one_hot_vector[8] = 1
        elif name == 'cow':
            one_hot_vector[9] = 1
        elif name == 'diningtable':
            one_hot_vector[10] = 1
        elif name == 'dog':
            one_hot_vector[11] = 1
        elif name == 'horse':
            one_hot_vector[12] = 1
        elif name == 'motorbike':
            one_hot_vector[13] = 1
        elif name == 'person':
            one_hot_vector[14] = 1
        elif name == 'pottedplant':
            one_hot_vector[15] = 1
        elif name == 'sheep':
            one_hot_vector[16] = 1
        elif name == 'sofa':
            one_hot_vector[17] = 1
        elif name == 'train':
            one_hot_vector[18] = 1
        elif name == 'tvmonitor':
            one_hot_vector[19] = 1
        elif name == 'container':
            one_hot_vector[20] = 1
        elif name == 'bucket':
            one_hot_vector[21] = 1
        elif name == 'wheelbarrow':
            one_hot_vector[22] = 1
        elif name == 'tire':
            one_hot_vector[23] = 1
        elif name == 'water':
            one_hot_vector[24] = 1   
        elif name == 'cattail':
            one_hot_vector[25] = 1 
        elif name == 'vegetation':
            one_hot_vector[26] = 1
        elif name == 'trash':
            one_hot_vector[27] = 1
        elif name == 'bird bath':
            one_hot_vector[28] = 1
        else:
            print('unknown label: %s' %name)

        return one_hot_vector

## example on how to use it
# import pickle
# data = XML_preprocessor('VOC2007/Annotations/').data
# pickle.dump(data,open('VOC2007.p','wb'))

