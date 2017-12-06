#original by rykov8
#https://github.com/rykov8/ssd_keras/blob/master/PASCAL_VOC/get_data_from_XML.py

import numpy as np
import os
from xml.etree import ElementTree
import pandas as pd

#CLASSES1 = ["n03991062", "n02802426", "container","water","bird bath", "tire", "wheelbarrow", "bucket", "gutter","vegetation","tree","building","sky""]

#CLASSES2 = ["background","pot","clutter","car","trash","woodpile","air conditioning","table","tire","fence","flower pot","porch","house","kayak","front yard","toy","bench","flower bed","driveway","yard","street","pool","chair","tarp","bird bath","container","back yard","gutter","trash bin","fire pit","air conditioner","umbrella","treehouse","basketball","tent","hose","drain","barbecue","awning","fountain","bucket","ladder","toy"]

#CLASSES3 = ["background","pot","clutter","car","trash","table","tire","flower pot","porch","kayak","toy","pool","chair","tarp","bird bath","container","gutter","trash bin","air conditioner","umbrella","basketball","tent","hose","drain","awning","bucket","ladder"]

#CLASSES4 = ["background","pot","flower pot","tarp","umbrella","basketball","trash bin","porch","container","gutter","trash bin","toy"]

CLASSES5 = ["background","pot","flower pot","tarp","umbrella","basketball","trash bin","porch","container","gutter","trash bin","toy","kayak","drain","clutter"]

CLASSES6 = ["background","pot","flower pot","tarp","umbrella","basketball","trash bin","porch","container","gutter","trash bin","toy","kayak","drain","clutter","car","table","recycling bin","bird bath","larvae","no larvae"]

CLASSES7 = ["background","pot","flower pot","tarp","umbrella","basketball","trash bin","porch","container","gutter","trash bin","trash","toy","kayak","drain","clutter","car","table","recycling bin","bird bath","larvae","no larvae", "air conditioner", "stone feature", "watering can","ladder"]

CLASSES = CLASSES7

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
                #print("width =",width)
                height = float(size_tree.find('height').text)
                #print("height =",height)
                #if width == 0:
                    #print("Res error for file", filename)
                #if height == 0:
                    #print("Res error for file", filename)
                n=0 #ssd breaks when training data has zero bounding boxes; this initiattes a count
                for object_tree in root.findall('object'):
                    for bounding_box in object_tree.iter('bndbox'):
                        n+=1 #count
                        xmin = float(bounding_box.find('xmin').text)/width
                        ymin = float(bounding_box.find('ymin').text)/height
                        xmax = float(bounding_box.find('xmax').text)/width
                        ymax = float(bounding_box.find('ymax').text)/height
                    bounding_box = [xmin,ymin,xmax,ymax]
                    bounding_boxes.append(bounding_box)
                    class_name = object_tree.find('name').text
                    if class_name=='garbage bin':
                        class_name='trash bin'
                    #elif class_name=='flower pot':
                    #    class_name=='pot'
                    elif class_name=='air conditioning':
                        class_name=='air conditioner'
                    elif class_name=='flower pt':
                        class_name=='flower pot'
                    elif class_name in ["continer","cotnainer"]:
                        class_name=='container'
                    elif class_name=="waterin gcan":
                        class_name=='watering can'
                    one_hot_class = self._to_one_hot(class_name)
                    one_hot_classes.append(one_hot_class)
                image_name = root.find('filename').text
                if image_name[-4:] != '.JPG':
                    image_name = image_name + '.JPG'
                if n == 0:
                    print("don't forget to remove", image_name, "it has zero bounding boxes") # print image names with zero bounding boxes for removal
                bounding_boxes = np.asarray(bounding_boxes)
                one_hot_classes = np.asarray(one_hot_classes)
                image_data = np.hstack((bounding_boxes, one_hot_classes))
                self.data[image_name] = image_data
                
    def return_bbox_coords(self):
        
        filenames = os.listdir(self.path_prefix)
        keys = [f[:-4] for f in filenames]
        boxes = pd.DataFrame(index=[keys,CLASSES],cols=['n','xmin','ymin','width','height'])
        
        for filename in filenames:
            if filename.lower().endswith('.xml'):
                tree = ElementTree.parse(self.path_prefix + filename)
                root = tree.getroot()
                bounding_boxes = []
                one_hot_classes = []
                size_tree = root.find('size')
                width = float(size_tree.find('width').text)
                height = float(size_tree.find('height').text)
                n=0 #ssd breaks when training data has zero bounding boxes; this initiattes a count
                for object_tree in root.findall('object'):
                    for bounding_box in object_tree.iter('bndbox'):
                        n+=1 #count
                        xmin = float(bounding_box.find('xmin').text)
                        ymin = float(bounding_box.find('ymin').text)
                        xmax = float(bounding_box.find('xmax').text)
                        ymax = float(bounding_box.find('ymax').text)
                    class_name = object_tree.find('name').text
                    if class_name=='garbage bin':
                        class_name='trash bin'
                    elif class_name=='flower pot':
                        class_name=='pot'
                    elif class_name=='air conditioning':
                        class_name=='air conditioner'

                    boxes.loc[(image_name,class_name),'n']=n
                    boxes.loc[(image_name,class_name),'xmin']=xmin
                    boxes.loc[(image_name,class_name),'ymin']=ymin
                    boxes.loc[(image_name,class_name),'width']=width
                    boxes.loc[(image_name,class_name),'height']=height
                    
        return bounding_boxes
                
                

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name in CLASSES:
            i = CLASSES.index(name)
            one_hot_vector[i] = 1
        #else:
        #    print(name)
        return one_hot_vector
    

## example on how to use it
# import pickle
# data = XML_preprocessor('VOC2007/Annotations/').data
# pickle.dump(data,open('VOC2007.p','wb'))

