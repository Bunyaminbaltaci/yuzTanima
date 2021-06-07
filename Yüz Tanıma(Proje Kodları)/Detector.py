import cv2
from time import sleep
from PIL import Image 

def main_app(name):
        # Hazır veri seti okunur.
        face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

        # OpenCV yüz tanımlayıcı aktif edilir.
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # kullanıcı resimleri okunur.
        recognizer.read(f"./data/classifiers/{name}_classifier.xml")

        # Video görüntüsü açılır
        cap = cv2.VideoCapture(0)
        pred = 0
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:


                roi_gray = gray[y:y+h,x:x+w]

                id,confidence = recognizer.predict(roi_gray)
                confidence = 100 - int(confidence)
                pred = 0
                if confidence > 50:
                            pred += +1
                            text = name.upper()
                            font = cv2.FONT_HERSHEY_PLAIN
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

                else:   
                            pred += -1
                            text = "Bilinmeyen Yuz"
                            font = cv2.FONT_HERSHEY_PLAIN
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

            cv2.imshow("Cikmak icin 'Q' tusuna basin...", frame)


            if cv2.waitKey(20) & 0xFF == ord('q'):
                print(pred)
                if pred > 0 : 
                    dim =(124,124)
                    img = cv2.imread(f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                    Image1 = Image.open(f".\\2.png") 
                      
                    # Orjinal görüntünün etkilenmemesi için bir kopyası oluşturulur.
                    Image1copy = Image1.copy() 
                    Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg") 
                    Image2copy = Image2.copy() 
                      
                    # Resmi yeni boyutuyla yapıştır. 
                    Image1copy.paste(Image2copy, (195, 114)) 
                      
                    # resmi kaydeder. 
                    Image1copy.save("end.png") 
                    frame = cv2.imread("end.png", 1)

                    cv2.imshow("Sonuclar",frame)
                    cv2.waitKey(5000)
                break


        cap.release()
        cv2.destroyAllWindows()
        
