import cv2
import os

filepath = './7eng_parking.MOV'
savepath = './data/test_img/'
video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

if not video.isOpened():
  print("Could not Open :", filepath)
  exit(0)

count = 0
i=0
while(video.isOpened()):
  ret, image = video.read()
  if(i==3):
    # image = cv2.flip(image, 0)
    # image = cv2.flip(image, 1)
    count_str=str(count)
    name = count_str.zfill(5)
    cv2.imwrite(savepath + "frame" + name + ".jpg", image)
    print('Saved frame number :', str(int(video.get(1))))
    count += 1
    i=0

  else:
    i=i+1

video.release()