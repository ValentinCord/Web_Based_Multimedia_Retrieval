import os

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method == "POST":

        uploaded_file = request.files['input_file']
        if uploaded_file.filename != '':
            os.makedirs(os.path.join('img_loaded'), exist_ok=True)
            uploaded_file.save(os.path.join('img_loaded', secure_filename(uploaded_file.filename)))
            
        top = request.form.getlist("top")
        rgb_hist = request.form.getlist("rgb_hist")
        hsv_hist = request.form.getlist("hsv_hist")
        ORB = request.form.getlist("ORB")
        SIFT = request.form.getlist("SIFT")
        VGG16 = request.form.getlist("VGG16")
        RESNet50 = request.form.getlist("RESNet50")

        print('top', top)
        print('rgb_hist', rgb_hist)
        print('hsv_hist', hsv_hist)
        print('ORB', ORB)
        print('SIFT', SIFT)
        print('VGG16', VGG16)
        print('RESNet50', RESNet50)

        return redirect(url_for("index"))

    return render_template('index.html')

if __name__ == '__main__':
    app.run()