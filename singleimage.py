from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import argparse
import PIL
from calorie import calories
from test1 import *

model=load_model("output/Pmobilenetmedium.hdf5")
data=[]
image=[]

labels={0:'chicken biryani',1:'bisibelebath',2:'butternaan',3:'chaat',4:'chappati',5:'dhokla',6:'masala dosa',7:'gulabjamun',8:'halwa',9:'idly',10:'kathi roll',11:'mendu vada',12:'noodles',13:'paniyaram',14:'poori',15:'samosa',16:'tandoori chicken',17:'upma',18:'vada pav',19:'ven pongal'}
path="data/evaluation/butternaan/naantest (15).jpg"
img=PIL.Image.open(path)
img=img.resize((256,256))
image.append(img)
img=np.asarray(img,dtype=np.float32)
img=img/255.0
img=img.reshape(-1,256,256,3)
pred=model.predict(img)
#print(pred)
w=max(max(pred))
pred=np.argmax(pred)
#print(max(max(pred)))
name=labels[pred]
print(labels[pred]," = ",w*100,"%")

x(path,pred)
