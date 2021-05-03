from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField,BooleanField, SelectField
from wtforms.validators import DataRequired,EqualTo,Email,Length,ValidationError
from face_recognition.database import TimeTable,user, faculty


class UserRegistrationForm(FlaskForm):
    username = StringField(label='Username',
                           validators=[DataRequired(), Length(min=2, max=20)], render_kw={"placeholder " : "Username"})
    email = StringField(label='Email', validators=[DataRequired(), Email()],render_kw={"placeholder " : "Email"})
    password = PasswordField(label='Password', validators=[DataRequired(), Length(min=2, max=20)],render_kw={"placeholder " : "Password"})
    confirmpassword = PasswordField(label='Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')], render_kw={"placeholder " : "Confirm Password"})
    submit = SubmitField(label='Sign Up')

    def validate_username(self,username):
        entry = user.query.filter_by(username=username.data).first()
        if entry:
            raise ValidationError('Already registered !')

    def validate_email(self,email):
        entry = user.query.filter_by(email=email.data).first()
        if entry:
            raise ValidationError('Already registered !')


class FacultyRegistrationForm(FlaskForm):
    username = StringField(label='Username',
                           validators=[DataRequired(), Length(min=2, max=20)], render_kw={"placeholder " : "Username"})
    name = StringField("Name", validators=[DataRequired(), Length(2,50)], render_kw={"placeholder " : "Name"})
    email = StringField(label='Email', validators=[DataRequired(), Email()],render_kw={"placeholder " : "Email"})
    password = PasswordField(label='Password', validators=[DataRequired(), Length(min=2, max=20)],render_kw={"placeholder " : "Password"})
    confirmpassword = PasswordField(label='Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')], render_kw={"placeholder " : "Confirm Password"})
    submit = SubmitField(label='Sign Up')

    def validate_username(self,username):
        entry = faculty.query.filter_by(username=username.data).first()
        if entry:
            raise ValidationError('Already registered !')

    def validate_email(self,email):
        entry = faculty.query.filter_by(email=email.data).first()
        if entry:
            raise ValidationError('Already registered !')

class UserLoginForm(FlaskForm):

    email = StringField(label='Email', validators=[DataRequired(), Email()],render_kw={"placeholder " : "Email"})
    password = PasswordField(label='Password', validators=[DataRequired(),Length(min=2,max=20)],render_kw={"placeholder " : "Enter Password"})
    submit = SubmitField(label='Login')


class RegisterForm(FlaskForm):
    enrollment = StringField("Enrollment",validators=[DataRequired(),Length(12,12)])
    name = StringField("Name", validators=[DataRequired(), Length(2,50)])
    email = StringField("Email",validators=[DataRequired(),Email()])
    sem = StringField("Sem",validators=[DataRequired()])
    branch = StringField("Branch",validators=[DataRequired()])
    password = PasswordField("Password",validators=[DataRequired()])
    confirm_password = PasswordField("Confirm Password",validators=[DataRequired(),EqualTo("password")])
    submit = SubmitField("Sign Up")

    #validate enrollment & email using predefined validate_columnname() method
    def validate_enrollment(self, enrollment):
        student = user.query.filter_by(enroll=enrollment.data).first()
        if student:
            raise ValidationError('That enrollment is taken. Please choose a different one.')

    def validate_email(self, email):
        student = user.query.filter_by(email=email.data).first()
        if student:
            raise ValidationError('That email is taken. Please choose a different one.')


class AddTimeTableForm(FlaskForm):
    subject = StringField("Subject",validators=[DataRequired(),Length(2,50)])
    sem = StringField("Sem", validators=[DataRequired()])
    batch = StringField("Batch", validators=[DataRequired()])
    slot = StringField("Slot", validators=[DataRequired()])
    faculty_name = StringField("Faculty Name", validators=[DataRequired(), Length(2,50)])
    day = StringField("Day", validators=[DataRequired(), Length(2,50)])

    start_time = StringField("Start Time",validators=[DataRequired()])
    end_time = StringField("End Time", validators=[DataRequired()])
    submit = SubmitField("Add")

    #validate enrollment & email using predefined validate_columnname() method
    def validate_slot(self, slot):
        timetable = TimeTable.query.filter_by(slot=slot.data).first()
        if timetable:
            raise ValidationError('That slot is taken. Please choose a different one.')

    def validate_start_time(self, start_time):
        timetable = TimeTable.query.filter_by(start_time=start_time.data).first()
        if timetable:
            raise ValidationError('That start_time is taken. Please choose a different one.')

    def validate_end_time(self, end_time):
        timetable = TimeTable.query.filter_by(end_time=end_time.data).first()
        if timetable:
            raise ValidationError('That end_time is taken. Please choose a different one.')

class LoginForm(FlaskForm):
    email = StringField("Email",validators=[DataRequired(),Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    remember_me = BooleanField("remember me")
    submit = SubmitField("Log in")


class Search(FlaskForm):
    sem = SelectField("Semester", choices=['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'], validators=[DataRequired()])
    sub = SelectField("Subject", choices=[], validators=[DataRequired()])
    batch = SelectField("Division", choices = ['G', 'H'], validators=[DataRequired()])
    submit = SubmitField("Search Attendance")