# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 00:48:36 2020

@author: REZA
"""
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

#%%

datapath=['dataset/normal/*','dataset/covid/*']
 
print("I'm loading images !")


imagePaths = glob(datapath[0])
data = []
labels = []

#224 for all  
#inception: 299

imageSize= 224


for imagePath in tqdm(imagePaths):

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imageSize, imageSize))

    data.append(image)
    labels.append("normal")


imagePaths = glob(datapath[1])
for imagePath in tqdm(imagePaths):
    
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imageSize, imageSize))

    data.append(image)
    labels.append("covid")

print("Images loading is done!")

#%%
seed=np.random.seed()


data = np.array(data) / 255.0
labels = np.array(labels)


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
#[0,1] normal
#[1,0] covid
labels = to_categorical(labels)
classes=lb.classes_

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=seed)


#%%

trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")


#%%
#VGG19

from vgg19Model import Build_VGG19
History_VGG19,predIdxs_model_vgg19 = Build_VGG19(trainX, trainY,testX, testY,trainAug,labels,classes)


fpr_keras_19, tpr_keras_19, thresholds_keras_19 = roc_curve(testY.argmax(axis=1), predIdxs_model_vgg19)
auc_keras_19 = auc(fpr_keras_19, tpr_keras_19)

#%%
#VGG16

from vgg16Model import Build_VGG16
History_VGG16,predIdxs_model_vgg16 = Build_VGG16(trainX, trainY,testX, testY,trainAug,labels,classes)


fpr_keras_16, tpr_keras_16, thresholds_keras_16 = roc_curve(testY.argmax(axis=1), predIdxs_model_vgg16)
auc_keras_16 = auc(fpr_keras_16, tpr_keras_16)

#%%
#ResNet50
from Resnet50Model import Build_ResNet50
History_Res,predIdxs_model_ResNet50 = Build_ResNet50(trainX, trainY,testX, testY,trainAug,labels,classes)


fpr_ResNet50, tpr_ResNet50, thresholds_ResNet50= roc_curve(testY.argmax(axis=1), predIdxs_model_ResNet50)
auc_ResNet50 = auc(fpr_ResNet50, tpr_ResNet50)

#%%
#inception v3
#input size of 299-by-299
#so we go to resize images
# The suffle data
from InceptionV3Model import Build_inceptionv3
History_inceptionv3,predIdxs_model_inception_v3 = Build_inceptionv3(trainX, trainY,testX, testY,trainAug,labels,classes)


fpr_inception_v3, tpr_inception_v3, thresholds_inception_v3 = roc_curve(testY.argmax(axis=1), predIdxs_model_inception_v3)
auc_inception_v3 = auc(fpr_inception_v3, tpr_inception_v3)


#%%
plt.figure()
plt.plot([0, 1], [0, 1])

plt.plot(fpr_keras_16, tpr_keras_16, label='VGG16 (area = {:.3f})'.format(auc_keras_16))

plt.plot(fpr_keras_19, tpr_keras_19, label='VGG19 (area = {:.3f})'.format(auc_keras_19))

plt.plot(fpr_inception_v3, tpr_inception_v3, label='Inception V3 (area = {:.3f})'.format(auc_inception_v3))

plt.plot(fpr_ResNet50, tpr_ResNet50, label='ResNet50 (area = {:.3f})'.format(auc_ResNet50))


#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


#%%

# You sohuld do these before inception, because in that case
# we change image size and also we shuffle again, so the results obtained followng 
# functions can be refutable.
# in other words: testY will be changed by shuffling

########################
# History_VGG19
# History_VGG16
# History_Res  
# History_inceptionv3
###########################

print(History_inceptionv3.history.keys())
plt.plot(History_inceptionv3.history['accuracy'])
plt.plot(History_inceptionv3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.plot(History_inceptionv3.history['loss'])
plt.plot(History_inceptionv3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
########################
# predIdxs_model_vgg19
# predIdxs_model_vgg16
# predIdxs_model_ResNet50  
# predIdxs_model_inception_v3
###########################


cm = confusion_matrix(testY.argmax(axis=1), predIdxs_model_inception_v3 )

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt='',xticklabels=lb.classes_,yticklabels=lb.classes_)

#sns.heatmap(cm.T,square=True,annot=True,fmt='d',cbar=False,
#            xticklabels=lb.classes_,yticklabels=lb.classes_)



    
    
    
    