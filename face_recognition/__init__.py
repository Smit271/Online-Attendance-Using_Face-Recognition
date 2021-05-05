from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from keras.models import load_model
import tensorflow as tf
global graph
import pickle
import cv2

#to hide tf warnings
tf.logging.set_verbosity(tf.logging.ERROR)
app = Flask(__name__)

#scret key setup
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

#loading models
model_path = "./face_recognition/WebCam_Face_Recognition/models/"

print("[FACENET Model] is loading...")
model = load_model(model_path+'facenet_keras.h5')
app.config['FACENET_MODEL'] = model
graph = tf.get_default_graph()
app.config['GRAPH'] = graph
print("[FACENET Model] is loaded!")

# pretrained deep learning model for face detection
print("[CAFFE Model] is loading...")
caffe_model = cv2.dnn.readNetFromCaffe(model_path + "deploy.prototxt.txt",
                                 model_path + "res10_300x300_ssd_iter_140000.caffemodel")
if caffe_model:
    print("[CAFFE] model loaded")
    app.config['CAFFE_MODEL'] = caffe_model


# face classifier SVM model
print("[SVM model] is loading...")
svm_model = pickle.load(open(model_path + "svm_train_face_model", 'rb'))
label_encoder = pickle.load(open(model_path + "label_train_face_encoder", "rb"))
app.config['SVM_MODEL'] = svm_model
app.config['LABEL_ENCODER'] = label_encoder
print("[SVM model] is loaded")
# load label encoder

#DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

db = SQLAlchemy(app)

#from face_recognition.WebCam_Face_Recognition import modules
from face_recognition import routes