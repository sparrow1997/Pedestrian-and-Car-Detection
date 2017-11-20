
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


people_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')


# In[15]:


cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    gray  = img#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    people = people_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in people:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        
    cars = car_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
     
    cv2.imshow('Haar_Cascade',img)
    
    if cv2.waitKey(30) & 0xff == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
        
        

        
        


# In[13]:


gray

