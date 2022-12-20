import os

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from time import time
from recherche import *

app = Flask(__name__)

app.secret_key = "super secret key"

cfg = {
    'descriptors' : {},
    'distance' : {},
    'input' : {},
    'result' : {}
}

@app.route("/", methods = ['GET', 'POST'])
def index():

    if request.method == "POST" and "form_config" in request.form:
        get_descriptor_form()
        get_distance_form()

    if request.method == 'POST' and "form_input_image" in request.form:
        get_input_form()

    if request.method == 'POST' and "form_search" in request.form:
        img_path = cfg['input']['img_path']
        descriptors = [k for k, v in cfg['descriptors'].items() if v == True and k != 'is_selected']
        distance = cfg['distance']['name']

        start = time()
        result = recherche(img_path, descriptors, distance)
        print(result)

        cfg['result']['time'] = round(time() - start, 3)
        cfg['result']['names'] = result


    if request.method == 'POST' and "form_top_20" in request.form:
        cfg['aff_20'] = True
        cfg['aff_50'] = False

    if request.method == 'POST' and "form_top_50" in request.form:
        cfg['aff_20'] = False
        cfg['aff_50'] = True

    if request.method == 'POST' and "form_rp" in request.form:
        pass

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
            flash('Pas de descripteur sélectionné')
            return redirect(request.url)

def get_distance_form():
    cfg['distance']['is_selected'] = False
    cfg['distance']['name'] = request.form.get('distance')
    cfg['distance']['is_selected'] = False if cfg['distance']['name'] is None else True

    if not cfg['distance']['is_selected']:
        flash('Pas de distance sélectionnée')
        return redirect(request.url)

def get_input_form():
    cfg['input']['is_selected'] = False
    cfg['input']['img_path'] = None

    if 'file' not in request.files:
        flash('Pas de fichier')
        return redirect(request.url)

    file = request.files['file']
    filename = file.filename
    allowed_file = '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

    if file.filename == '':
        flash('Pas de fichier sélectionné')
        return redirect(request.url)

    elif allowed_file:
        os.makedirs(os.path.join('static', 'img_loaded'), exist_ok = True)
        file.save(os.path.join('static', 'img_loaded', secure_filename(file.filename)))
        cfg['input']['is_selected'] = True
        cfg['input']['img_path'] = "static/img_loaded/" + file.filename

    else:
        flash('Erreur: Format de fichier non accepté, veuillez mettre un png/jpg/jpeg')
  
if __name__ == "__main__":
    app.run(debug=True)