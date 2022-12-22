import os

from flask import Flask, render_template, request, redirect, flash

from werkzeug.utils import secure_filename
from time import time

from recherche import *
from mongo import Mongo

app = Flask(__name__)
mongo = Mongo()

app.secret_key = 'mykey'

cfg = {
    'descriptors' : {},
    'distance' : {},
    'input' : {},
    'result' : {},
    'show' : {},
    'vector' : ['BGR', 'HSV', 'GLCM', 'HOG', 'LBP'],
    'matrix' : ['SIFT', 'ORB']
}

@app.route("/", methods = ['GET', 'POST'])
def main():

    if request.method == 'POST' and 'form_config' in request.form:
        get_descriptor_form()
        get_distance_form()

    if request.method == 'POST' and "form_input_image" in request.form:
        get_input_form()

    if request.method == 'POST' and "form_search" in request.form:
        img_path = cfg['input']['img_path']
        descriptors = [k for k, v in cfg['descriptors'].items() if v == True and k != 'is_selected']
        distance = cfg['distance']['name']

        start = time()
        result = recherche(mongo, img_path, descriptors, 'euclidean', 'flann', cfg)

        cfg['result']['time'] = round(time() - start, 3)
        cfg['result']['names'] = result
        cfg['result']['done'] = True

    if request.method == 'POST' and "form_top_20" in request.form:
        cfg['show']['20'] = True
        cfg['show']['50'] = False
        cfg['show']['rp'] = False

    if request.method == 'POST' and "form_top_50" in request.form:
        cfg['show']['20'] = False
        cfg['show']['50'] = True
        cfg['show']['rp'] = False

    if request.method == 'POST' and "form_rp" in request.form:
        cfg['show']['20'] = False
        cfg['show']['50'] = False
        cfg['show']['rp'] = True

    return render_template("index.html", cfg = cfg)

def get_descriptor_form():
    cfg['descriptors']['is_selected'] = False
    cfg['descriptors']['SIFT'] = True if request.form.get('SIFT') != None else False
    cfg['descriptors']['BGR']  = True if request.form.get('BGR')  != None else False
    cfg['descriptors']['GLCM'] = True if request.form.get('GLCM') != None else False
    cfg['descriptors']['HOG']  = True if request.form.get('HOG')  != None else False
    cfg['descriptors']['HSV']  = True if request.form.get('HSV')  != None else False
    cfg['descriptors']['LBP']  = True if request.form.get('LBP')  != None else False
    cfg['descriptors']['ORB']  = True if request.form.get('ORB')  != None else False
    cfg['descriptors']['DL']   = True if request.form.get('DL')   != None else False
    cfg['descriptors']['is_selected'] = any(cfg['descriptors'].values())

    if not cfg['descriptors']['is_selected']:
        flash('[CONFIG] No Descriptor selected')
        return redirect(request.url)

def get_distance_form():
    cfg['distance']['is_selected'] = False
    cfg['distance']['name'] = request.form.get('distance')
    cfg['distance']['is_selected'] = False if cfg['distance']['name'] is None else True

    if not cfg['distance']['is_selected']:
        flash('[CONFIG] No distance selected')
        return redirect(request.url)

def get_input_form():
    cfg['input']['is_selected'] = False
    cfg['input']['img_path'] = None

    if 'file' not in request.files:
        flash('[FILE] No file selected')
        return redirect(request.url)

    file = request.files['file']
    filename = file.filename
    allowed_file = '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

    if file.filename == '':
        flash('[FILE] No file selected')
        return redirect(request.url)

    elif allowed_file:
        os.makedirs(os.path.join('static', 'img_loaded'), exist_ok = True)
        file.save(os.path.join('static', 'img_loaded', secure_filename(file.filename)))
        cfg['input']['is_selected'] = True
        cfg['input']['img_path'] = "static/img_loaded/" + file.filename

    else:
        flash('[FILE] Wrong file format')
  
if __name__ == "__main__":
    app.run(debug=True)