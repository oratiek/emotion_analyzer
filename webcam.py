import numpy as np
import cv2
import csv
import sys
import os
from datetime import datetime
from keras.preprocessing.image import img_to_array
from keras.models import load_model

class EmotionRecgnition:
	def __init__(self, root):
		self.root = root

	def logger(self, msg):
		print(datetime.now(), msg)

	def prep(self):
		foldername = "folder{}".format(len(os.listdir(self.root)))
		self.folder_path = os.path.join(self.root, foldername)
		self.images_path = os.path.join(self.root, foldername, "images")
		self.log_path = os.path.join(self.folder_path,"log.csv")
		# create folders and file
		os.mkdir(self.folder_path)
		os.mkdir(self.images_path)
		os.system("touch {}".format(self.log_path))
		#本当は一つひとつチェックした方がセキュアではあるが、基本的にエラーが出ないような構成なので今回は無視
		self.logger("all files has been created")

	def classify(self,img,model):
		img = cv2.resize(img,(48,48))
		img = img.astype("float32")/255.0
		img = img_to_array(img)
		img = np.expand_dims(img, axis=0)
		result = model.predict(img)[0]
		return result.argmax(), result

	def webCam(self):
		# initial settings
		model = load_model("model.h5")
		fontType = cv2.FONT_HERSHEY_SIMPLEX
		EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
		cascade = cv2.CascadeClassifier("cascade.xml")
		cap = cv2.VideoCapture(0)
		self.fps = round(cap.get(cv2.CAP_PROP_FPS))

		# open log file
		f = open(self.log_path,"a")
		writer = csv.writer(f)

		filenum = 0
		while True:
			ret, frame = cap.read()
			frame = cv2.flip(frame, 1)
			img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
			faces = cascade.detectMultiScale(img, 1.3, 5)
			for x,y,w,h in faces:
				face = img[y:y+h,x:x+w]
				emotion, prob = self.classify(face,model)
				saveData = np.append(prob, [EMOTIONS[emotion],datetime.now()])
				print(saveData)
				writer.writerow(saveData)
				cv2.putText(frame,EMOTIONS[emotion],(x,y), fontType, 3,(255,0,0),2,cv2.LINE_AA)
				cv2.putText(frame,"Angry:"+str(prob[0]),(0,100), fontType, 1,(255,0,0),2,cv2.LINE_AA)
				cv2.putText(frame,"Disgust:"+str(prob[1]),(0,200), fontType, 1,(255,0,0),2,cv2.LINE_AA)
				cv2.putText(frame,"Scared:"+str(prob[2]),(0,300), fontType, 1,(255,0,0),2,cv2.LINE_AA)
				cv2.putText(frame,"Happy:"+str(prob[3]),(0,400), fontType, 1,(255,0,0),2,cv2.LINE_AA)
				cv2.putText(frame,"Sad:"+str(prob[4]),(0,500), fontType, 1,(255,0,0),2,cv2.LINE_AA)
				cv2.putText(frame,"Surprised:"+str(prob[5]),(0,600), fontType, 1,(255,0,0),2,cv2.LINE_AA)
				cv2.putText(frame,"Neutral:"+str(prob[6]),(0,700), fontType, 1,(255,0,0),2,cv2.LINE_AA)
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)
			cv2.imshow("frame",frame)
			filepath = os.path.join(self.images_path, str(filenum) + ".jpg")
			print(filepath)
			cv2.imwrite(filepath, frame)
			filenum += 1
			if cv2.waitKey(1) == ord("q"):
				self.logger("recognition finished")
				self.logger("saving log file")
				f.close()
				self.logger("session closed")
				break

	def createVideo(self):
		self.logger("start creating video")
		output_path = os.path.join(self.folder_path,"result.mp4")
		command = "ffmpeg -r 10 -i {}/%d.jpg -vcodec libx264 -pix_fmt yuv420p -r 10 {}".format(self.images_path, output_path)
		os.system(command)
		self.logger("result.mp4 has been created")


if __name__ == "__main__":
	recognizer = EmotionRecgnition("webcam")
	recognizer.prep()
	recognizer.webCam()
	recognizer.createVideo()





