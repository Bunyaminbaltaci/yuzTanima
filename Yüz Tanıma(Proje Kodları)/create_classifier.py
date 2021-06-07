import numpy as np
from PIL import Image
import os, cv2



# Yüz tanıma işlemi yapan modeli eğitmek için oluşturulan fonksiyon.
def train_classifer(name):
    # custom veri setindeki tüm dosyalar okunur.
    path = os.path.join(os.getcwd()+"/data/"+name+"/")

    faces = []
    ids = []
    labels = []
    pictures = {}


    # Kaydedilen resimler numpy array formatına dönüştürülüp diziye aktarılır.

    for root,dirs,files in os.walk(path):
            pictures = files


    for pic in pictures :

            imgpath = path+pic
            img = Image.open(imgpath).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(pic.split(name)[0])
            faces.append(imageNp)
            ids.append(id)

    ids = np.array(ids)

    #Model eğitilir ve model vektörleri kaydedilir.
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("./data/classifiers/"+name+"_classifier.xml")

