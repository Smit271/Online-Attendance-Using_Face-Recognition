from face_recognition import app
print("face recognition called")
if __name__ == "__main__":
    #db.create_all()
    app.run(debug=True,port=5000)