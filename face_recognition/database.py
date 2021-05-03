from face_recognition import db,login_manager,bcrypt
from datetime import datetime
from flask_login import logout_user,LoginManager,UserMixin,login_required,login_user,current_user
from flask_bcrypt import Bcrypt

@login_manager.user_loader
def load_user(user_id):
    return user.query.get(int(user_id))


class Student(db.Model):
    date_time = str(datetime.now())

    id = db.Column(db.Integer, primary_key=True)
    enrollment = db.Column(db.String(12), unique=True, nullable=False)
    name = db.Column(db.String(20), unique=False, nullable=False)
    email = db.Column(db.String(50), unique=False, nullable=False)
    sem = db.Column(db.Integer, unique=False, nullable=False)
    branch = db.Column(db.String(20), nullable=False, default='Computer')
    div = db.Column(db.String(20), nullable=True)

    attendence = db.relationship("Attendence",backref="student")

    def __repr__(self):
        return f"Student('{self.name}','{self.branch}','{self.enrollment}')"


class user(db.Model,UserMixin):
    userid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True)
    email = db.Column(db.String(30), unique=True)
    type = db.Column(db.String(30), unique=False)
    password = db.Column(db.String(60), unique=False)
    confirmpassword = db.Column(db.String(60), unique=False)

    def __repr__(self):
        return f"user('{self.username}','{self.email}')"

    def verify_password(self,pwd):
        return bcrypt.check_password_hash(self.password,pwd)

    def is_active(self):
        return True

    def is_authenticated(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.userid)

    def get_email(self):
        return str(self.email)

class faculty(db.Model,UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True)
    name = db.Column(db.String())
    type = db.Column(db.String(30), unique=False)
    email = db.Column(db.String(30), unique=True)
    password = db.Column(db.String(60), unique=False)
    confirmpassword = db.Column(db.String(60), unique=False)
    faculty = db.relationship("TimeTable", backref="faculty")

    def __repr__(self):
        return f"faculty('{self.username}','{self.email}')"

    def verify_password(self,pwd):
        return bcrypt.check_password_hash(self.password,pwd)

    def is_active(self):
        return True

    def is_authenticated(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


class TimeTable(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(12), unique=False, nullable=False)
    sem = db.Column(db.Integer, unique=False, nullable=False)
    batch = db.Column(db.String(12), unique=False, nullable=False)
    slot = db.Column(db.Integer, unique=False, nullable=False)
    #faculty_name = db.Column(db.String(50), unique=False, nullable=False)
    faculty_id = db.Column(db.Integer, db.ForeignKey("faculty.id"))
    day = db.Column(db.String(50), unique=False, nullable=False)
    attendence = db.relationship("Attendence", backref="time_table")

    def get_id(self):
        return int(self.id)

    def __repr__(self):
        return f"Student('{self.subject}','{self.sem}','{self.slot}')"

class Attendence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(50), unique=False, nullable=False)
    time = db.Column(db.String(50), unique=False, nullable=False)
    student_id = db.Column(db.Integer,db.ForeignKey("student.id"))
    timetable_id = db.Column(db.Integer,db.ForeignKey("time_table.id"))
    def __repr__(self):
        return f"Student('{self.student_id}','{self.timetable_id}','{self.time}, '{self.date}')"

'''
command line commands:
>>> from face_recognition import db
>>> from face_recognition.database import Student
>>> db.create_all()
>>> s1 = Student(name="karm")
>>> db.session.add(s1)
>>> Student.query.all()
>>> Student.query.filter_by(name="karm").all()
>>> Student.query.filter_by(name="karm").first()
>>> Student.query.get(id)
'''