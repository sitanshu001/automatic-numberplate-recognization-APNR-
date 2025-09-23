import cv2
import easyocr
import matplotlib.pyplot as plt

image_path = "car.jpg"  
img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

reader = easyocr.Reader(['en'])
for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    roi = img[y:y + h, x:x + w]  

    result = reader.readtext(roi)
    print("Detected Text:", result)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
