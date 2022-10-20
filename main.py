import cv2
import face_recognition as fr

imgRod = fr.load_image_file('rodinei.jpg')
imgRod = cv2.cvtColor(imgRod, cv2.COLOR_BGR2RGB)

imgRodTest = fr.load_image_file('rodinei2.jpg')
imgRodTest = cv2.cvtColor(imgRodTest, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgRod)[0]
cv2.rectangle(imgRod, (faceLoc[3], faceLoc[0],
              faceLoc[1], faceLoc[2]), (0, 255, 0), 2)


encodeRod = fr.face_encodings(imgRod)[0]
encodeRodTest = fr.face_encodings(imgRodTest)[0]

comparing = fr.compare_faces([encodeRod], encodeRodTest)
distance = fr.face_distance([encodeRod], encodeRodTest)

print(comparing, distance)
cv2.imshow('imgRod', imgRod)
cv2.imshow('imgRodTest', imgRodTest)
cv2.waitKey(0)
