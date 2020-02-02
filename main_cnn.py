# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:42:46 2020

@author: josep
"""
from keras.layers import Dense, Lambda, MaxPooling2D, Flatten, Concatenate, BatchNormalization, Dropout, Multiply
from keras.models import Model, Input
from keras import optimizers
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.model_selection import train_test_split
import json
import numpy as np
from PIL import Image
import glob
import cv2


#fonction pour chercher les attribus en fonction de l'index de l'image
def get_test_attributes(index,test_train):
    i=market[test_train]['image_index'].index(index);
    #print(i)
    tmp_list=[]
    attri_list=market[test_train]
    for attribute in attri_list :
        if attribute != "image_index":
            if attribute=="age":
                tmp_list.append(market[test_train][attribute][i]/4)#pour normaliser entre 0 et 1
            else:
                tmp_list.append(market[test_train][attribute][i]-1)#pour normaliser entre 0 et 1
    return tmp_list 

#fonction pour retourner une image horizontalement
def horizontal_flip(image_array):
    return image_array[:, ::-1]

#importation demarket.json
with open('market_attribute.json') as json_data:
     market= json.load(json_data)
#market=np.array(market)

#importation de gallery.json
with open('gallery.json') as json_data:
     gallery= json.load(json_data)
gallery=np.array(gallery)




#importation des images
images=[]
train_images=[]
test_images=[]

attributs_market=[]
attributs_train=[]
attributs_test=[]

#listes des index pour le modele d'identification
image_index_train=[]
image_index_test=[]

#trie des images entre le test et le train
for filename in glob.glob('Market-1501\*.jpg'): #assuming gif
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    images.append(image)
    str_index=filename[12:16]
    #print(type(str_index))
    if str_index in market['test']['image_index']:
        #print("l'image est dans test")
        test_images.append(image)
        attributs_test.append(get_test_attributes(str_index,"test"))
        image_index_test.append(str_index)
    else:
        #print("l'image est dans train")
        train_images.append(image)
        attributs_train.append(get_test_attributes(str_index,"train"))
        image_index_train.append(str_index)

print(len(image_index_test))
print(len(image_index_train))
     
#data-augmentation(question3)     
#symétrie horizontale des images
augmented_train_images=[]
augmentes_train_attributs=[]
augmented_test_images=[]
augmentes_test_attributs=[]


for images in train_images:
    augmented_train_images.append(image)
    augmented_train_images.append(horizontal_flip(image))
    
for attri in  attributs_train:
    augmentes_train_attributs.append(attri)
    augmentes_train_attributs.append(attri)

for images in test_images:
    augmented_test_images.append(image)
    augmented_test_images.append(horizontal_flip(image))  

for attri in attributs_test:
    augmentes_test_attributs.append(attri)
    augmentes_test_attributs.append(attri)   

    
       
#Séparation x_train et y_train, X_test et Y_test
X_train=np.array(train_images)
X_test=np.array(test_images)
y_train=np.array(attributs_train)
y_test=np.array(attributs_test)

print(y_train)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)



#definissons l'input du modèle en fonction de la taille des images
input_tensor = Input(shape=(128, 64, 3))

#création du modèle pré-entrainé
model = ResNet50(input_shape=(128, 64, 3), pooling="avg", weights='imagenet',include_top=False)(input_tensor)

#création d"une nouvelle couche pour les attributs (ici il y a 27 attributs)
attributs = Dense(27, activation="softmax")(model)


#couche fully connected
fullyconnected = Dense(512, activation="relu")(attributs)
fullyconnected = Dropout(rate=0.5)(fullyconnected)

#il y a 1501 index différents dans le dataset
identification=Dense(1501,activation='softmax')(fullyconnected)


#Modele 1
#question 1: En utilisant un resnet50 pré-entrainé sur imagenet, implémentez l'algoritme de classification des attributs
model_Q1=Model(inputs=input_tensor, outputs=attributs)

#compilation du modèle
model_Q1.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

#entrainement
model_Q1.fit(x=X_train,y=y_train,epochs=20)

#resultats
res = model_Q1.evaluate(X_test,y_test )
print("evaluation du modèle :",res)



#Modele2 : Implementé un premier algo de ré identification avec unde couche fully connected supplémentaire pour l'identification
model_Q2=Model(inputs=input_tensor,outputs=identification)

#compilation du modèle
model_Q2.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


y_train=np.array(image_index_train)
y_train=to_categorical(y_train)#one hot encodage
y_train=np.array(y_train)

y_test=np.array(image_index_test)
y_test=to_categorical(y_test)#one hot encodage
y_test=np.array(y_test)

#entrainement
model_Q2.fit(x=X_train,y=y_train,epochs=20)


#resultats
res2 = model_Q2.evaluate(X_test,y_test)
print("evaluation du modèle :",res2)




