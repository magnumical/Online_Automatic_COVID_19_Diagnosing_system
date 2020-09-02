# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:39:26 2020

@author: REZA
"""



from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from time import strftime
import seaborn as sns
plt.style.use('seaborn-poster')



def Build_inceptionv3(trainX, trainY,testX, testY,trainAug,labels,classes):
    
    
    INIT_LR = 1e-3;EPOCHS = 10;BS = 8
    
    baseModel = InceptionV3(weights="imagenet", include_top=False,input_tensor=Input(shape=(299, 299, 3)))
    
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    
    
    model_inception_v3 = Model(inputs=baseModel.input, outputs=headModel)
    
    for layer in baseModel.layers:
        layer.trainable = False
    
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model_inception_v3.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
    
    print("[INFO] training head...")
    Historyinceptionv3 = model_inception_v3.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    
    
    print("[INFO] evaluating network...")
    predIdxs_model_inception_v3 = model_inception_v3.predict(testX, batch_size=BS)
    
    predIdxs_model_inception_v3 = np.argmax(predIdxs_model_inception_v3, axis=1)
    
    
    
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs_model_inception_v3)
    
    sns.heatmap(cm.T,square=True,annot=True,fmt='d',cbar=False,
                xticklabels=['Normal','Covid19'],yticklabels=['Normal','Covid19'] )
        
    print(classification_report(testY.argmax(axis=1), predIdxs_model_inception_v3))

    
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    
    print("[INFO] saving COVID-19 detector model...")
    model_inception_v3.save("./models/"+strftime("5.25.inception.h5",), save_format="h5")
    
    
    plt.figure()
    plt.plot(Historyinceptionv3.history['accuracy'])
    plt.plot(Historyinceptionv3.history['val_accuracy'])
    plt.plot(Historyinceptionv3.history['loss'])
    plt.plot(Historyinceptionv3.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy / loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'Validation accuracy','loss','Validation loss'])
    plt.show()
    return Historyinceptionv3,predIdxs_model_inception_v3
    