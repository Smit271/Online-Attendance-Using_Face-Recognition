# import the necessary packages
import traceback

from imutils.video import VideoStream
from flask_sqlalchemy import SQLAlchemy
from flask_login import current_user
from face_recognition.database import Attendence,user,TimeTable, Student, faculty
from face_recognition import db,app
from face_recognition.WebCam_Face_Recognition import modules
from datetime import datetime
import numpy as np
from PIL import Image
from numpy import asarray
import imutils
import pickle
import cv2
from scipy import ndimage
from pathlib import Path
#from mtcnn.mtcnn import MTCNN

def camera():

	prototxt = "deploy.prototxt.txt"
	confidence_threshold = 0.9
	required_size = (160,160)
	filepath = "./face_recognition/8-Dataset/"
	path = ""
	model_path = "./face_recognition/WebCam_Face_Recognition/models/"
	#model_path = "./WebCam_Face_Recognition/models/"
	db.create_all()

	#pretrained facenet model to gate embeddings of face
	facenet_model = app.config['FACENET_MODEL']
	#facenet_model = modules.load_facenet_model()

	#pretrained deep learning model for face detection
	dnn_model = modules.load_caffe_model()

	#face classifier SVM model
	svm_model = pickle.load(open(model_path+"svm_train_face_model", 'rb'))

	#load label encoder
	label_encoder =  pickle.load(open(model_path+"label_train_face_encoder", "rb"))

	#svm_model, label_encoder = modules.load_face_classifier()
	pre_confidence = 0.5

	#ip webcam
	#url = "http://192.168.0.101:8080/video"
	#url = "http://192.168.43.1:8080/video"

	# load our serialized model from disk
	# initialize the video stream and allow the cammera sensor to warmup
	print("starting video stream...")

	#capture from PC's Camera
	#vs = VideoStream(src=0).start()

	#Capture from Phone's Camera
	vs = cv2.VideoCapture(0)
	if not vs:
		print("camera can not capture frame!")
		return 0

	#store already present students
	slot = int(modules.get_slot())
	today_date = str(datetime.today().date())
	#tt_slot = TimeTable.get_id()
	temp_id = current_user.get_email()
	faculty_id = faculty.query.filter_by(email = temp_id).first().id
	print(slot)
	print(faculty_id)
	try :
		timetable_details = TimeTable.query.filter_by(slot = slot, faculty_id = faculty_id, date = today_date).first()
		timetable_id = timetable_details.id

	except:
		timetable_details = TimeTable.query.filter_by(faculty_id = faculty_id, date = today_date).first()
		timetable_id = timetable_details.id

	print(timetable_id)
	today_data = Attendence.query.filter_by(timetable_id = timetable_id, date = str(datetime.today().date()))
	print(today_data.first())
	attendence_taken = []
	if today_data:
		for each in today_data.all():
			student_id = each.id
			student_email = user.query.filter_by(userid=student_id).first().email
			student_name = Student.query.filter_by(email = student_email).first().name
			attendence_taken.append(student_name)
	print(attendence_taken)
	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels

		#for mobile camera
		h,frame = vs.read()
		#
		frame = imutils.resize(frame, width=360)
		#for pc camera
		#frame = vs.read()
		#frame = imutils.resize(frame, width=720)
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		#image preprocessing like mean substraction to pass it in neural networks
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

		#blob.shape = (1, 3, 300, 300)

		# pass the blob through the network and obtain the detections and
		# predictions
		dnn_model.setInput(blob)
		detections = dnn_model.forward()

		#detections = mtcnn_model.detect_faces(frame)
		#detection.shape =  (1, 1, 200, 7)
		'''
		7 columns:
	
		3rd column = confidence of pizel
		4th column = (startX)/width
		5th column = (startY)/height
		6th column = (endX)/width
		7th column = (endY)/height
	
		'''
		# loop over the detections
		#for each in detections:
		for i in range(0, detections.shape[2]):

			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]
			#confidence = each['confidence']

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence

			if confidence < pre_confidence:
				continue

			# compute the (x, y)-coordinates of the bounding box for the
			# object

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

			startX, startY, endX, endY = box.astype("int")


			# draw the bounding box of the face along with the associated
			# probability
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
			#fix y position,where we have to put text
			y = startY - 10 if startY - 10 > 10 else endY + 10
			y_f = endY + 10 if endY + 10 < h else startY - 10

			#crop face from image array
			crop = frame[startY:endY, startX:endX]



			try:
				#convert array into image
				crop_img = Image.fromarray(crop)
				# resize face to (160,160,3)
				crop_img = crop_img.resize(required_size)
				#convert face image into array
				face = asarray(crop_img)
				#get embeddings from image
				embs = modules.get_embeddings(facenet_model, face)
				#to normalize face pixels
				encoder = modules.Normalizer()
				embs = encoder.transform(modules.expand_dims(embs, axis=0))

				#predict class lebel in numeric data type
				class_val = svm_model.predict(embs)
				#convert numeric label into string i.e. 0 -> karm
				class_name = label_encoder.inverse_transform(class_val)[0]
				#confidence to recognize
				class_probability = max(svm_model.predict_proba(embs)[0])

				if class_probability > 0.45 and class_name not in attendence_taken:
					#get lecture slot depending upon present time
					slot = str(modules.get_slot())
					#get student name
					#print(class_name)
					t1 = TimeTable.query.filter_by(slot = slot, date = today_date, faculty_id = faculty_id).first()
					s1 = Student.query.filter_by(name = class_name, sem = t1.sem).first()
					print(s1)
					s1_enroll = s1.enrollment
					print(s1_enroll)
					'''s1_sem = s1.sem
					s1_div = s1.batch
					tt_sem = timetable_details.sem
					tt_batch = timetable_details.batch
					print(s1_sem)
					print(s1_div)
					print(tt_sem)
					print(tt_batch)'''
					#get TimeTable id
					#A = Attendence.query.filter_by(student_id = s1.id,dtimetable_id=t1.id).first()
					#add in Attendence Table
					print(slot)
					print(s1,t1)
					if s1 and t1:
						date = str(datetime.today().date())
						time = str(datetime.now().time())
						A1 = Attendence(date = date, time = time, student_id=s1.id, timetable_id=t1.id)
						#print(s1,t1,A1)
						db.session.add(A1)
						db.session.commit()
						attendence_taken.append(class_name)
				# confidence %
				text = str(class_name) + "-"+ str(round(class_probability * 100, 2)) +"%"
				text_face = "f= " + str(round(confidence,2))
				# put text above rectangle
				cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
				cv2.putText(frame, text_face, (startX, y_f), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
			except Exception:
				traceback.print_exc()
				continue

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1)

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()

	#for pc camera
	#vs.stop()

	vs.release()
	print("camera stopped")
	#& 0xFF
