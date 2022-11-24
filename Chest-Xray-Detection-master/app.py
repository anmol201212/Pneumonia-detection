import model

import os
from urllib import response
from flask import Flask, flash, request, redirect, url_for, Response, render_template, send_from_directory
import flask
from werkzeug.utils import secure_filename
ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
# from tensorflow.keras import models

# seg_model = models.load_model("/content/drive/MyDrive/unet_resnet34_body_parse.h5")


UPLOAD_FOLDER = ROOT_DIR + "/static/upload"
DWNLD_FOLDER = ROOT_DIR + "/static/gen"
ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DWNLD_FOLDER"] = DWNLD_FOLDER

@app.route("/upload", methods = ["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return "no file1"
        file = request.files["file"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == "":
            flash("No selected file")
            return "no file2"
        if file:
            filename = secure_filename(file.filename)
            # change file name to 1.jpg
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], '1.jpg'))
    return render_template("infer.html")

@app.route("/infer", methods = ["GET", "POST"])
def get_result():
    if request.method == "GET":
        if request.args.get("sr") == "true":
            pth = model.load_output(ROOT_DIR)  
 
        else:
            pth = model.load_output(ROOT_DIR)
    
    if pth == 0:
        # send output as normal
        # make json packet
        return render_template('negative.html')
    else:
        # send output as pneumonia
        return render_template('positive.html')



@app.route("/")
def main():
    return render_template("index.html")

if __name__ == "__main__":
    app.secret_key = "super secret key"
    app.config["SESSION_TYPE"] = "filesystem"
    app.run()