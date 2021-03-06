import cv2
import os
import numpy as np
cap = cv2.VideoCapture(0)

directory = os.path.abspath('1')[:-1] 

# directory = '/Users/user/Desktop/face_recognition/'
dirlist = os.listdir(directory)
a = True 
for i in dirlist:
    if i == 'images':
        a = False
        
if a == True:
    os.mkdir('images')

data_path = directory + 'images/'
onlyfiles = os.listdir(data_path)

Training_Data, Labels = [],[]

if onlyfiles == []: 
    name = raw_input("Input your name: ")

    path = data_path + name + "/"
   
    os.mkdir(path)    
    i = 0
    while i <= 45:
        ret, frame = cap.read()
        face_xml = cv2.CascadeClassifier('face.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_xml.detectMultiScale(gray, 1.3, 15)
        if len(faces)==1:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_color = gray[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(path, str(i) + ".jpg"), roi_color)
        i+=1


for i, files in enumerate(onlyfiles):
        file_path = data_path +onlyfiles[i]
        if onlyfiles[i] == ".DS_Store":
            continue
        onlyimages = os.listdir(file_path)
        for j, files in enumerate(onlyimages):
            image_path = file_path + "/" + onlyimages[j]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)

model = cv2.face.LBPHFaceRecognizer_create()

model.train( np.asarray(Training_Data) ,np.asarray(Labels))

def nothing(x):
  pass
def reg(frame):
    
    name = raw_input("Input your name: ")
    path = data_path + name + "/"
    os.mkdir(path)    
    i = 0
    while i <= 45:
        ret, frame = cap.read()
        face_xml = cv2.CascadeClassifier('face.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_xml.detectMultiScale(gray, 1.3, 15)
        if len(faces)==1:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_color = gray[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(path, str(i) + ".jpg"), roi_color)
        i+=1

while (True): 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('face.xml')
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face:
        final = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lab = model.predict(final[y:y+h, x:x+w])
        if lab[1] < 40:
            name=onlyfiles[lab[0]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(33) == ord('a'):
        reg(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
cap.release()
cv2.destroyAllWindows()
