import os

from flask import Flask, render_template, request, redirect, flash, session
from werkzeug.utils import secure_filename
from time import time

from recherche import *
from mongo import Mongo

# Flask app initialization
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_KEY')

# Mongo initialization
mongo = Mongo()

# Configuration initialization
cfg = {
    'descriptors' : {
        'is_selected' : False,
    },
    'distance' : {
        'is_selected' : False,
    },
    'input' : {
        'is_selected' : False,
    },
    'result' : {},
    'show' : {},
    'vector' : ['BGR', 'HSV', 'GLCM', 'HOG', 'LBP', 'VGG16', 'XCEPTION', 'MOBILENET'],
    'matrix' : ['SIFT', 'ORB'],
    'metrics' : {
        'classe': {},
        'subclasse': {}
    }
}

@app.route('/', methods = ['GET', 'POST'])
def main():

    # if not logged in, redirect to login page
    if not session.get('logged_in'):
        return render_template('login.html')

    # parse descriptor form
    if request.method == 'POST' and 'form_desc' in request.form:
        get_descriptor_form()
    
    # parse distance form
    if request.method == 'POST' and 'form_dist' in request.form:
        get_distance_form()

    # parse input form
    if request.method == 'POST' and 'form_input_image' in request.form:
        get_input_form()

    # indexation of the query image and search for best matches
    if request.method == 'POST' and 'form_search' in request.form:

        # check if all required fields are selected
        if not (cfg['descriptors']['is_selected'] and cfg['distance']['is_selected'] and cfg['input']['is_selected']):
            flash('[CONFIG] Missing configuration')
            return redirect(request.url)

        # get all parameters
        img_path = cfg['input']['img_path']
        descriptors = [k for k, v in cfg['descriptors'].items() if v == True and k != 'is_selected']
        distance_vect = cfg['distance']['vect']
        distance_matrix = cfg['distance']['matrix']

        # launch indexation search
        start = time()
        result = recherche(mongo, img_path, descriptors, distance_vect, distance_matrix, cfg)
        
        # save results
        cfg['result']['time'] = round(time() - start, 3)
        cfg['result']['names'] = result
        cfg['result']['done'] = True

        # if image query is in the database, analyze metrics
        if img_path.split('/')[-1] in os.listdir('static/db/'):
            save_metrics(cfg, mongo)

    # parse top 20 form to only show top 20 results
    if request.method == 'POST' and 'form_top_20' in request.form:
        cfg['show']['20'] = True
        cfg['show']['50'] = False
        cfg['show']['rp'] = False

    # parse top 50 form to only show top 50 results
    if request.method == 'POST' and 'form_top_50' in request.form:
        cfg['show']['20'] = False
        cfg['show']['50'] = True
        cfg['show']['rp'] = False

    # parse rp form to only show rp results
    if request.method == 'POST' and 'form_rp' in request.form:
        cfg['show']['20'] = False
        cfg['show']['50'] = False
        cfg['show']['rp'] = True

    return render_template('index.html', cfg = cfg)


@app.route('/login', methods = ['POST'])
def login():
    if request.method == 'POST' and 'form_login' in request.form:
        # check if credentials are correct
        # if yes, set session to logged in
        if request.form.get('username') == 'admin' and request.form.get('password') == 'admin':
            session['logged_in'] = True
        else:
            flash('[LOGIN] Wrong credentials')
    return main()

@app.route('/logout')
def logout():
    # set session to logged out
    session['logged_in'] = False
    return main()

@app.route('/history')
def history():
    # return history page
    return render_template('history.html', mongo = mongo)

@app.route('/help')
def help():
    # return help page
    return render_template('help.html')


def get_descriptor_form():
    cfg['descriptors']['is_selected'] = False

    # get status of checkboxes for descriptors
    cfg['descriptors']['SIFT']          = True if request.form.get('SIFT')      != None else False
    cfg['descriptors']['BGR']           = True if request.form.get('BGR')       != None else False
    cfg['descriptors']['GLCM']          = True if request.form.get('GLCM')      != None else False
    cfg['descriptors']['HOG']           = True if request.form.get('HOG')       != None else False
    cfg['descriptors']['HSV']           = True if request.form.get('HSV')       != None else False
    cfg['descriptors']['LBP']           = True if request.form.get('LBP')       != None else False
    cfg['descriptors']['ORB']           = True if request.form.get('ORB')       != None else False
    cfg['descriptors']['VGG16']         = True if request.form.get('VGG16')     != None else False
    cfg['descriptors']['XCEPTION']      = True if request.form.get('XCEPTION')  != None else False
    cfg['descriptors']['MOBILENET']     = True if request.form.get('MOBILENET') != None else False

    # check if at least one descriptor is selected
    cfg['descriptors']['is_selected'] = any(cfg['descriptors'].values())

    # if no descriptor selected, flash error
    if not cfg['descriptors']['is_selected']:
        flash('[CONFIG] No Descriptor selected')
        return redirect(request.url)

def get_distance_form():
    cfg['distance']['is_selected'] = False

    # get selected distance for vector and matrix
    cfg['distance']['vect'] = request.form.get('distance_vect')
    cfg['distance']['matrix'] = request.form.get('distance_matrix')

    # check if at least one distance is selected for each category
    cfg['distance']['is_selected'] = False if cfg['distance']['vect'] is None or cfg['distance']['vect'] is None else True

    # if distance is not selected, flash error
    if not cfg['distance']['is_selected']:
        if cfg['distance']['vect'] is None:
            flash('[CONFIG] No vector distance selected')
        if cfg['distance']['matrix'] is None:
            flash('[CONFIG] No matrix distance selected')
        return redirect(request.url)

def get_input_form():
    cfg['input']['is_selected'] = False
    cfg['input']['img_path'] = None

    # check if image is uploaded
    if 'file' not in request.files:
        flash('[FILE] No file selected')
        return redirect(request.url)

    # parse uploaded file
    file = request.files['file']
    filename = file.filename
    allowed_file = '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

    # check if file is selected
    if file.filename == '':
        flash('[FILE] No file selected')
        return redirect(request.url)

    # check if file is allowed
    elif allowed_file:
        # save file to static/img_loaded
        os.makedirs(os.path.join('static', 'img_loaded'), exist_ok = True)
        file.save(os.path.join('static', 'img_loaded', secure_filename(file.filename)))
        cfg['input']['is_selected'] = True
        cfg['input']['img_path'] = 'static/img_loaded/' + file.filename

    # if file is not allowed, flash error
    else:
        flash('[FILE] Wrong file format')
  
if __name__ == '__main__':
    app.run(debug=True)