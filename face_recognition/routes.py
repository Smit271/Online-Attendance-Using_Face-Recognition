from flask import Flask, render_template, Blueprint, request, redirect, url_for, current_app,flash,session,logging, session, send_file, make_response
from flask import send_from_directory
from face_recognition import app,db,bcrypt
from face_recognition.database import faculty,TimeTable,Attendence,user, Student
from face_recognition.forms import RegisterForm,AddTimeTableForm, UserRegistrationForm, UserLoginForm, FacultyRegistrationForm, Search
from flask_login import logout_user,LoginManager,UserMixin,login_required,login_user,current_user
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from face_recognition.WebCam_Face_Recognition import modules
from flask_bcrypt import Bcrypt
import bcrypt
from face_recognition.WebCam_Face_Recognition import LiveFaceRecognition
from datetime import datetime
import numpy as np
import datetime as dt
from functools import wraps
from sqlalchemy import and_
from reportlab.platypus import SimpleDocTemplate, Table
from reportlab.lib.pagesizes import letter 
import pandas as pd

#face images folder
app.config['UPLOAD_FOLDER'] = "./face_recognition/8_5-Dataset/train"
#app.config['PDF_LOC'] = "/home/smit/Smit/Projects/06_Face_Recognition/Attendance-using-face-recognition/Attendance_pdf"
app.config['PDF_LOC'] = os.path.join(os.getcwd(), 'Attendance_pdf')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'heic', 'jpeg', 'gif',''}

def allowed_file(file):
    temp = file[::-1]
    if '.' not in file:
        return False
    extension = temp[:temp.index('.')][::-1]
    print(extension)
    if extension.lower() in ALLOWED_EXTENSIONS:
        return True
    return False

def student_login_required(page):
    @wraps(page)
    def wrap(*args,**kwargs):
        if current_user.is_authenticated:
            if current_user.type == "student":
                return page(*args,**kwargs)
            else:
                flash("you need to login first as student!", "danger")
                logout_user()
                return redirect(url_for("userlogin"))
        else:
            flash("you need to login first","danger")
            return redirect(url_for("userlogin"))
    return wrap

def faculty_login_required(page):
    @wraps(page)
    def wrap(*args,**kwargs):
        if current_user.is_authenticated:
            if current_user.type == "faculty":
                return page(*args,**kwargs)
            else:
                flash("you need to login first as faculty!", "danger")
                logout_user()
                return redirect(url_for("login_faculty"))
        else:
            flash("you need to login first","danger")
            return redirect(url_for("userlogin"))
    return wrap

@app.route("/")
@app.route("/home")
def home():
    db.create_all()
    #print(data)
    return render_template("index.html")

@app.route("/faculty-home")
def facultyhome():
    db.create_all()
    return render_template("Admin/index.html")

@app.route("/add-lecture", methods=["POST","GET"])
@faculty_login_required
def timetable():
    #print(current_user)
    today_date = str(datetime.now().date())
    #print(today_date)
    #form  = AddTimeTableForm()
    if request.method == "POST":
        sub = request.form["subject"]
        name = request.form["first_name"]
        sem = request.form["sem"]
        batch = request.form["div"]
        slot = request.form["slot"][0]

        #temp_id = current_user.get_id()
        #print(temp_id) --> Gives faculty id of user tabel
        temp_email = current_user.get_email()
        faculty_id = faculty.query.filter_by(email = temp_email).first().id

        # Checking of that slot by any faculty on that day is taken or not
        tt_taken = TimeTable.query.filter(and_(TimeTable.slot == slot,
                                            TimeTable.sem == sem,
                                            TimeTable.batch == batch,
                                            TimeTable.date == today_date)).first()
        
        #print(tt_taken)
        if tt_taken:
            taken_faculty_name = faculty.query.filter_by(id = tt_taken.faculty_id).first().name
            flash(f"Today's slot {slot} already taken for {sem} - {batch} for Subject {tt_taken.subject} by {taken_faculty_name} !!", "danger")
            return redirect(url_for("timetable"))


        #print(faculty_id)

        if sum == "" or name == "" or sem == "" or batch=="" or slot=="":
            flash("enter all data","danger")
            return redirect(url_for("timetable"))

        entry = TimeTable(subject = sub, sem = sem, batch=batch,slot=slot,
                          faculty_id=faculty_id, date = today_date)
        #db.create_all()
        db.session.add(entry)
        try:
            db.session.commit()
        except:
            flash("Database is locked")
            db.session.rollback()
            #db.session.close()
        flash("Data added", "success")
        return redirect(url_for("timetable"))

    return render_template("Admin/add_lecture.html")

@app.route("/student-total-attendance", methods=["GET","POST"])
@student_login_required
def student_total_attendance():
    #count total days
    start_term = dt.date(2021, 4, 20)
    today = dt.date(2021,5,10)
    end_term = min(datetime.today().date(),today)
    total_days = np.busday_count(start_term, end_term)+1

    #get student id, student name & batch
    user_email = current_user.get_email()
    student_id = Student.query.filter_by(email = user_email).first()
    #email = user.query.filter_by(userid=user_id).first().email
    #student_id = Student.query.filter_by(email = email).first()

    if not student_id:
        flash("Can't open Attendance Page, First upload your pictures here!","danger")
        return redirect(url_for("add_photos"))

    student_id = student_id.id

    #get Student data
    student = Student.query.filter_by(id = student_id).first()
    name = student.name
    sem = student.sem
    batch = student.sem + "-" + student.div
    user_data = {"name": name, "batch": batch,"start_term":start_term,"total":total_days}

    #getting all lectures are which are attended
    attended_lecs = Attendence.query.filter_by(student_id = student_id).all()

    #count total attendance of particular subject
    attended_lecs_ids = {}
    for each in attended_lecs:
        try:
            attended_lecs_ids[each.timetable_id] += 1
        except:
            attended_lecs_ids[each.timetable_id] = 1

    print(attended_lecs_ids)
    row = []

    #TimeTable rows
    lecs = TimeTable.query.filter_by(sem = sem).all()

    for lec in lecs:
        temp = {"subject":lec.subject,"slot":lec.slot,"present":0}

        #note total attendance
        if lec.id in attended_lecs_ids:
            temp['present'] = attended_lecs_ids[lec.id]
        row.append(temp)

    print(user_data)
    print(row)

    '''
    user_data.keys= name,batch
    row.keys = slot,subject,time,status
    '''


    return render_template("total_attendance.html",row=row,user_data=user_data)

@app.route("/student-today-attendance", methods=["GET","POST"])
@student_login_required
def student_today_attendance():
    today_date = str(datetime.today().date())
    #get student id, student name & batch
    user_email = current_user.get_email()
    #email = user.query.filter_by(userid=user_id).first().email
    student_details = Student.query.filter_by(email = user_email).first()
    if not student_details:
        flash("Can't open Attendance Page, First upload your pictures here!","danger")
        return redirect(url_for("add_photos"))
    student_id = student_details.id
    student = Student.query.filter_by(id = student_id).first()
    name = student.name
    sem = student.sem
    batch = str(sem) + "-" + str(student.div)
    user_data = {"name": name, "batch": batch}

    #getting which lectures are attended
    attended_lecs = Attendence.query.filter_by(student_id = student_id,date=today_date).all()
    attended_lecs_ids = [each.timetable_id for each in attended_lecs]
    print(attended_lecs_ids)
    row = []

    #TimeTable rows
    lecs = TimeTable.query.filter_by(sem = sem).all()

    for lec in lecs:
        temp = {"subject":lec.subject, "slot":lec.slot,"time":"-","status":"absent"}
        if lec.id in attended_lecs_ids:
            time = Attendence.query.filter_by(student_id=student_id,timetable_id=lec.id).first().time
            temp['time'] = time
            temp["status"] = "PRESENT"
        row.append(temp)

    print(user_data)
    print(row)

    '''
    user_data.keys= name,batch
    row.keys = slot,subject,time,status
    '''
    return render_template("attendance.html",row=row,user_data=user_data)

@app.route("/faculty-attendance", methods=["GET","POST"])
@faculty_login_required
def faculty_attendance():
    today_date = str(datetime.today().date())

    temp_mail = current_user.get_email()
    faculty_id = faculty.query.filter_by(email = temp_mail).first().id
    print(f"faculty id:{faculty_id}")
    row = TimeTable.query.filter_by(faculty_id=faculty_id).distinct().all()
    print(row)
    form  = Search()
    sub = [row[i].subject for i in range(len(row))]
    subs = list(set(sub))
    form.sub.choices = subs
    if request.method == "POST":
        sub = request.form['sub']
        div = request.form['div']
        sem = request.form['sem']
        print(sub, div, sem)
        row = TimeTable.query.filter_by(faculty_id=faculty_id, sem = sem, batch = div, subject = sub).all()
        print(row)

        if not row:
            flash(f"Can't Find Lecture of {sem} - {div} for Subject {sub} for today","danger")
            #return redirect(url_for("timetable"))
            return redirect(url_for("faculty_attendance"))
        #get subject of current faculty
        
        subjects = []
        #get time table id of that subject
        tt_id = row[0].id
        present_students = Attendence.query.filter_by(date=today_date,timetable_id=tt_id).all()
        #print(present_students)

        if not present_students:
            flash(f"You hadn't taken Any Attendance for Students of {sem} - {div} for Subject {sub} Today","danger")

        detail = {"subject":sub,"date":today_date}
        data = []
        for each in present_students:
            s = Student.query.filter_by(id = each.student_id).first()
            t = TimeTable.query.filter_by(id = each.timetable_id).first()
            temp = {"time":each.time,
            "name" : s.name,
            "enrollment" : s.enrollment,
            "subject" : t.subject,
            "batch" : t.batch,
            "slot" : t.slot
            }
            data.append(temp)
            #student_user_id = user.query.filter_by(email=student.email).first().userid
            #present_count = len(Attendence.query.filter_by(timetable_id=tt_id,student_id=student_user_id).all())
            #print(student_user_id,present_count)
            #temp['present'] = present_count
        #print(data0)
        return render_template("Admin/attendance.html", data = data, form = form, detail = detail)

    return render_template("Admin/attendance.html", form = form)

@app.route("/faculty-total-attendance", methods=["GET","POST"])
@faculty_login_required
def faculty_total_attendance():
    # count total days
    temp_mail = current_user.get_email()
    faculty_id = faculty.query.filter_by(email = temp_mail).first().id
    row = TimeTable.query.filter_by(faculty_id=faculty_id).distinct().all()

    form  = Search()
    # Finding Distinct Subject list 
    sub = [row[i].subject for i in range(len(row))]
    subs = list(set(sub))
    form.sub.choices = subs

    start_term = dt.date(2021, 4, 20)
    today = dt.date(2021, 5, 10)
    end_term = min(datetime.today().date(), today)
    today_date = str(datetime.today().date())
    total_days = np.busday_count(start_term, end_term)+1
    data0 = [["Name", "Enrollment", "Division", "Semester", "Total_Attended"]]
    #print(today_date)

    

    '''temp_mail = current_user.get_email()
    faculty_id = faculty.query.filter_by(email = temp_mail).first().id
    row = TimeTable.query.filter_by(faculty_id=faculty_id).all()'''

    if request.method == "POST":
        if request.form.get('search') == 'search':
        
            sub = request.form['sub']
            div = request.form['div']
            sem = request.form['sem']
            #print(sub, div, sem)
            row = TimeTable.query.filter_by(faculty_id=faculty_id, sem = sem, batch = div, subject = sub).all()

            if not row:
                flash(f"Can't Find Lecture of {sem} - {div} for Subject {sub}","danger")
                return redirect(url_for("timetable"))
            #get subject of current faculty
            data = []
            subjects = []
            for i in range(len(row)):
                subjects.append(row[i].subject)

                #get time table id of that subject
                tt_id = row[i].id
                lecture_detail = {"subject":subjects,"total":total_days,"start_term":start_term}
                student_rows = Student.query.filter_by(div = div, sem = sem).all()

                if not student_rows:
                    flash(f"You hadn't taken Any Attendance for Students of {sem} - {div} for Subject {sub}","danger")
                
                for student in student_rows:
                    temp = {"name":student.name,"enrollment":student.enrollment,
                            "batch": student.div,
                            "slot" : row[i].slot,
                            "sem" : student.sem,
                            "sub" : row[i].subject}
                    student_user_id = student.id
                    #student_user_id = user.query.filter_by(email=student.email).first().userid
                    present_count = len(Attendence.query.filter_by(timetable_id=tt_id,student_id=student_user_id).all())
                    #print(student_user_id,present_count)
                    temp['present'] = present_count
                    data.append(temp)
                #print(data0)
                return render_template("Admin/total_attendance.html", data = data, lecture_detail = lecture_detail, form = form)

        # If Faculty wants to create pdf by class- division - subject student's attendance
        elif request.form.get('pdf') == 'get pdf':
            sub = request.form['sub']
            div = request.form['div']
            sem = request.form['sem']
            #print(sub, div, sem)
            row = TimeTable.query.filter_by(faculty_id=faculty_id, sem = sem, batch = div, subject = sub).all()

            if not row:
                flash(f"Can't Find Lecture of {sem} - {div} for Subject {sub}","danger")
                return redirect(url_for("timetable"))
            #get subject of current faculty
            subjects = []
            
            for i in range(len(row)):
                subjects.append(row[i].subject)

                #get time table id of that subject
                
                lecture_detail = {"subject":subjects,"total":total_days,"start_term":start_term}
                student_rows = Student.query.filter_by(div = div, sem = sem).all()

            if not student_rows:
                flash(f"You hadn't taken Any Attendance for Students of {sem} - {div} for Subject {sub}","danger")
            
            for i in range(len(student_rows)):
                tt_id = row[0].id
                temp0 = []
                temp0.append(student_rows[i].name)
                temp0.append(student_rows[i].enrollment)
                temp0.append(student_rows[i].div)
                temp0.append(student_rows[i].sem)
                student_user_id = student_rows[i].id
                #student_user_id = user.query.filter_by(email=student.email).first().userid
                present_count = len(Attendence.query.filter_by(timetable_id=tt_id,student_id=student_user_id).all())
                temp0.append(present_count)
                data0.append(temp0)

            # Making PDF
            filename = f"{today}_{sub}_{data0[1][3]}_{data0[1][2]}_{row[0].faculty_id}.pdf"
            
            #files = os.getcwd()
            #print(files)
            
            
            file_name = app.config['PDF_LOC'] + f"/{filename}"
            modules.make_pdf(data0, file_name)
            print(file_name)
            try:
                return send_from_directory(app.config['PDF_LOC'], filename=filename, as_attachment=True)
            except FileNotFoundError:
                abort(404)
            response = make_response(file_name)
            response.headers['Content-Disposition'] = f"attachment; filename={filename}.pdf"
            response.mimetype = 'application/pdf'
            return response

        # If Faculty wants to create pdf by class- division - subject student's attendance
        elif request.form.get('excel') == 'get excel':
            sub = request.form['sub']
            div = request.form['div']
            sem = request.form['sem']
            #print(sub, div, sem)
            row = TimeTable.query.filter_by(faculty_id=faculty_id, sem = sem, batch = div, subject = sub).all()

            if not row:
                flash(f"Can't Find Lecture of {sem} - {div} for Subject {sub}","danger")
                return redirect(url_for("timetable"))

            enrs = [] # To store available enrollments of searched semester and division
            all_date = []# To store Dates of month
            final_list = []
            enr = Student.query.filter_by(sem = sem, div = div).all()
            for i in range(len(enr)):
                enrs.append(enr[i].enrollment)

            # Making Whole Month dates range to fill attendance
            month = today_date.split('-')[1]
            if (int(month) % 2 != 0) or (month == '08'):
                datelist = pd.date_range(datetime.today().replace(day=1), periods=31).tolist()
            elif (int(month) % 2 == 0):
                datelist = pd.date_range(datetime.today().replace(day=1), periods=30).tolist()
            
            # Putting into one list 
            all_date.append('Enrollment/Date')
            for j in range(len(datelist)):
                all_date.append(str(datelist[j].date()))

            # Making dates as First row
            final_list.append(all_date)
            print(enrs)


            for i in range(len(enrs)):
                i_atd = []
                i_atd.append(enrs[i])
                
                for j in range(1, len(all_date)):

                    
                    tt_id = TimeTable.query.filter_by(faculty_id=faculty_id,
                                                    sem = sem,
                                                    batch = div,
                                                    subject = sub,
                                                    date = all_date[j]).first()
                    print(tt_id)

                    if tt_id:
                        s_id = Student.query.filter_by(enrollment = enrs[i]).first().id
                        print(s_id)
                        print(tt_id.id)

                        atd_details = Attendence.query.filter_by(student_id = s_id, timetable_id = tt_id.id, date = all_date[j]).first()
                        print(atd_details)

                         # -- > Making Enrollment as First Column of every row
                        if atd_details:
                            i_atd.append(1)
                        else :
                            i_atd.append(0)
                    else :
                        i_atd.append(0)

                final_list.append(i_atd)
            print(len(final_list))
            print(final_list)

            # Making Excel File name
            filename = f"{today_date}_{sub}_{sem}_{div}.xlsx"
            
            #files = os.getcwd()
            #print(files)
            
            
            file_name = app.config['PDF_LOC'] + f"/{filename}"
            modules.make_excel(final_list, file_name)
            print(file_name)
            try:
                return send_from_directory(app.config['PDF_LOC'], filename=filename, as_attachment=True)
            except FileNotFoundError:
                abort(404)
            response = make_response(file_name)
            response.headers['Content-Disposition'] = f"attachment; filename={filename}.pdf"
            response.mimetype = 'application/pdf'
            return response
            #return render_template("Admin/total_attendance.html", form = form)

        return render_template("Admin/total_attendance.html", data = data, lecture_detail = lecture_detail, form = form)
    return render_template("Admin/total_attendance.html", form = form)

@app.route("/user-register",methods=["GET", "POST"])
def userregister():
    if current_user.is_authenticated and current_user.type == "student":
        return redirect(url_for('home'))
    form = UserRegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.hashpw(form.password.data.encode('utf-8'),bcrypt.gensalt())
        hashed_confirmpassword = bcrypt.hashpw(form.confirmpassword.data.encode('utf-8'), bcrypt.gensalt())
        #hashed_confirmpassword = bcrypt.generate_password_hash(form.confirmpassword.data).decode('utf-8')
        entry = user(username = form.username.data, email = form.email.data, type="student", password = hashed_password, confirmpassword = hashed_confirmpassword )
        db.session.add(entry)
        db.session.commit()
        flash('Your account has been created! Welcome Home !', 'success')
        return redirect(url_for("home"))
    return render_template('user-registration.html',form = form, title='UserRegister')

@app.route("/faculty-register",methods=["GET", "POST"])
def facultyregister():
    #print("faculty called!")
    if current_user.is_authenticated and current_user.type == "faculty":
        return redirect(url_for('facultyhome'))
    form = FacultyRegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.hashpw(form.password.data.encode('utf-8'),bcrypt.gensalt())
        hashed_confirmpassword = bcrypt.hashpw(form.confirmpassword.data.encode('utf-8'), bcrypt.gensalt())
        #hashed_confirmpassword = bcrypt.generate_password_hash(form.confirmpassword.data).decode('utf-8')
        entry = faculty(username = form.username.data, email = form.email.data, type="faculty", name = form.name.data, password = hashed_password, confirmpassword = hashed_confirmpassword )
        db.session.add(entry)
        db.session.commit()

        entry2 = user(username = form.username.data, email = form.email.data, type = "faculty", password = hashed_password, confirmpassword = hashed_confirmpassword)
        db.session.add(entry2)
        db.session.commit()

        flash('Your account has been created! Welcome Home !', 'success')
        return redirect(url_for("facultyhome"))
    return render_template('Admin/faculty-registration.html',form = form, title='FacultyRegister')


@app.route("/add_photos", methods = ["GET","POST"])
@student_login_required
def add_photos():
    student_user_id = current_user.userid
    email = user.query.filter_by(userid = student_user_id).first().email
    student = Student.query.filter_by(email = email).first()
    #print(student_user_id,email,student)
    if student:
        flash("Can't open:You have already added photos","danger")
        return redirect(url_for("home"))
    if request.method == "POST":
        name = request.form["name"]
        enroll = request.form["enroll"]
        email = request.form["email"]
        branch = request.form["branch"]
        sem = request.form["sem"]
        div = request.form["div"]

        #print(sem)

        if name == "" or enroll == "" or branch == "Select the branch" or sem == "Select your semester":
            flash("Enter all Details", "danger")
            return render_template("profile.html")


        student = Student(enrollment=enroll, name=name, email=email,
                          sem=sem,
                          branch=branch,div=div)
        files_objects = []
        # folder name = student name
        folder_label = name
        valid_upload = 1
        # iterate throgh all images
        for number in range(1, len(request.files) + 1):
            file = request.files['img' + str(number)]
            if not file.filename:
                flash(f"Please Upload Image--{number}", "danger")
                valid_upload = 0
                break
            # elif not allowed_file(file.filename):
            #     flash(f"please upload image file in Image-{number}", "danger")
            #     valid_upload = 0
            elif file:
                path = os.path.join(app.config['UPLOAD_FOLDER'], folder_label, secure_filename(file.filename))
                files_objects.append([file, path])
                # file.save(path)

        # create a folder in train dataset & save 5 images there
        if valid_upload:
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_label)
            # make folder of student
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            # save all images at given directory
            for file_path in files_objects:
                file_path[0].save(file_path[1])

            # save details in database
            # db.create_all()
            db.session.add(student)
            db.session.commit()

            flash("Data added successfully","success")
            return redirect(url_for("home"))

    return render_template("profile.html")





@app.route("/contact-us", methods = ["GET","POST"])
def contact_us():
    return render_template("contact-us.html")

@app.route("/faculty-contact-us", methods = ["GET","POST"])
def faculty_contact_us():
    return render_template("Admin/contact-us.html")

@app.route("/forget-password", methods = ["GET","POST"])
def forget_password():
    return render_template("forget-password.html")

@app.route("/login-user", methods=["GET","POST"])
def userlogin():
    if current_user.is_authenticated and current_user.type == "student":
        return redirect(url_for('home'))
    form = UserLoginForm()
    if form.validate_on_submit():
        entry = user.query.filter_by(email=form.email.data).first()
        if entry and entry.verify_password(form.password.data) and entry.type=="student":
            login_user(entry)
            return redirect(url_for('home'))
            flash('Successfully logged in !')
        else:
           flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login-user.html', form=form, title='UserLogin')

@app.route("/logout-user",methods=["GET","POST"])
@student_login_required
def logoutUser():
    logout_user()
    return redirect(url_for('home'))

@app.route("/logout-faculty",methods=["GET","POST"])
@faculty_login_required
def logoutFaculty():
    logout_user()
    return redirect(url_for('facultyhome'))

#############
#ADMIN Routes
#############

@app.route("/login-faculty", methods = ["GET","POST"])
def login_faculty():
    if current_user.is_authenticated and current_user.type == "faculty":
        return redirect(url_for('facultyhome'))
    form = UserLoginForm()
    if form.validate_on_submit():
        entry = user.query.filter_by(email=form.email.data).first()
        print(entry)
        if entry and entry.verify_password(form.password.data) and entry.type == "faculty":
            login_user(entry)
            return redirect(url_for('facultyhome'))
            flash('Successfully logged in !')
        else:
           flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template("Admin/login-faculty.html",form=form, title='UserLogin')



@app.route("/login-admin", methods = ["GET","POST"])
def login_admin():
    return render_template("login-admin.html")

@app.route("/update-model", methods = ["GET","POST"])
@faculty_login_required
def update_model():
    print(os.getcwd())
    labels = modules.add_new_persons()
    #LiveFaceRecognition.camera()
    return render_template("Admin/update model.html",labels=labels)

@app.route("/camera", methods = ["GET","POST"])
@faculty_login_required
def on_camera():
    print(os.getcwd())
    #modules.add_new_persons()
    LiveFaceRecognition.camera()
    return render_template("Admin/index.html")
