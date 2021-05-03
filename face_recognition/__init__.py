from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from keras.models import load_model
import tensorflow as tf
global graph
app = Flask(__name__)
model_path = "./face_recognition/WebCam_Face_Recognition/models/"
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

model = load_model(model_path+'facenet_keras.h5')
#model.keras_model._make_predict_function()

app.config['FACENET_MODEL'] = model

graph = tf.get_default_graph()
app.config['GRAPH'] = graph
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

db = SQLAlchemy(app)
db.create_all()
from face_recognition.WebCam_Face_Recognition import modules
from face_recognition import routes
