import cv2
import numpy as npy
import face_recognition as face_rec

def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

vijimg1 = face_rec.load_image_file('images\iviji3.jpg')
vijimg1 = cv2.cvtColor(vijimg1, cv2.COLOR_BGR2RGB)
vijimg1 = resize(vijimg1, 0.50)
vijimg2=face_rec.load_image_file('images\iviji4.jpg')
vijimg2 = cv2.cvtColor(vijimg2, cv2.COLOR_BGR2RGB)
vijimg2 = resize(vijimg2, 0.50)

faceLocation_viji1 = face_rec.face_locations(vijimg1)[0]
encode_viji1 = face_rec.face_encodings(vijimg1)[0]
cv2.rectangle(vijimg1, (faceLocation_viji1[3], faceLocation_viji1[0]), (faceLocation_viji1[1], faceLocation_viji1[2]), (255, 0, 255), 3)


faceLocation_viji2 = face_rec.face_locations(vijimg2)[0]
encode_viji2 = face_rec.face_encodings(vijimg2)[0]
cv2.rectangle(vijimg2, (faceLocation_viji2[3], faceLocation_viji2[0]), (faceLocation_viji2[1], faceLocation_viji2[2]), (255, 0, 255), 3)

print(encode_viji1)
print(encode_viji2)

results = face_rec.compare_faces([encode_viji1], encode_viji2)
print(results)
cv2.putText(vijimg2, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', vijimg1)
cv2.imshow('test_img1',vijimg2)
cv2.waitKey(0)
cv2.destroyAllWindows()