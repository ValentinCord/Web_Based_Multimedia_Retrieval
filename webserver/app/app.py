import os
import hashlib
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from time import time
from recherche import *
from mongo import Mongo

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_KEY')

mongo = Mongo()

cfg = {'descriptors' : {'is_selected' : True, 'XCEPTION_false_rmac' : True},
       'distance' : {'is_selected' : True, 'vect' : 'euclidean', 'matrix' : 'bruteForceMatching'},
       'input' : {'is_selected' : False,},
        'result' : {},
        'show' : {},
        'vector' : ['BGR', 'HSV', 'GLCM', 'HOG', 'LBP', 'VGG16_false', 'XCEPTION_false', 'MOBILENET_false', 'XCEPTION_true', 'VGG16_false_pca', 'XCEPTION_false_pca', 'MOBILENET_false_pca', 'XCEPTION_true_pca', 'VGG16_false_rmac', 'XCEPTION_false_rmac', 'MOBILENET_false_rmac', 'XCEPTION_true_rmac'],
        'matrix' : ['SIFT', 'ORB'],
        'metrics' : {'classe': {},'subclasse': {}}}

@app.route('/', methods = ['GET', 'POST'])
def main():

    # if not logged in, redirect to login page
    if not cfg.get('logged_in'):
        return render_template('login.html')

    if request.method == 'POST' and 'form_desc' in request.form:
        get_descriptor_form()
    if request.method == 'POST' and 'form_dist' in request.form:
        get_distance_form()
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

        start = time()
        result = recherche(mongo, img_path, descriptors, distance_vect, distance_matrix, cfg) 
        cfg['result']['time'] = int(time() - start) # time in seconds
        cfg['result']['names'] = result # list of names of the ordered best matches
        cfg['result']['done'] = True  # search is done

        # if image query is in the database, analyze metrics
        if cfg['input']['is_in_database']:
            save_metrics(cfg, mongo)

    if request.method == 'POST' and 'form_top_20' in request.form:
        cfg['show']['20'] = True
        cfg['show']['50'] = False
        cfg['show']['rp'] = False
    if request.method == 'POST' and 'form_top_50' in request.form:
        cfg['show']['20'] = False
        cfg['show']['50'] = True
        cfg['show']['rp'] = False
    if request.method == 'POST' and 'form_rp' in request.form:
        cfg['show']['20'] = False
        cfg['show']['50'] = False
        cfg['show']['rp'] = True

    return render_template('index.html', cfg = cfg)


@app.route('/login', methods = ['POST'])
def login():
    if request.method == 'POST' and 'form_register' in request.form:
        return render_template('register.html')

    if request.method == 'POST' and 'form_login' in request.form:
        user = request.form.get('username')
        pwd = request.form.get('password')
        hash_pwd = hashlib.md5(pwd.encode())
        if mongo.users.find_one({'username': user, 'password': hash_pwd.hexdigest()}):
            cfg['logged_in'] = True
        else:
            flash('[LOGIN] Wrong credentials')
    return redirect(url_for('main'))

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST' and 'form_register' in request.form:
        user = request.form.get('username')
        pwd = request.form.get('password')
        pwd_again = request.form.get('password_again')
        if pwd != pwd_again:
            flash('Wrong password')
            return render_template('register.html')
        # elif mongo.users.find_one({'username': user}):
        #     flash('User already exists')
        #     return render_template('register.html')
        else:
            hash_pwd = hashlib.md5(pwd.encode())
            mongo.users.insert_one({'username': user, 'password': hash_pwd.hexdigest()})
            cfg['logged_in'] = True
    return redirect(url_for('main'))

@app.route('/logout')
def logout():
    cfg['logged_in'] = False
    return main()

@app.route('/history')
def history():
    return render_template('history.html', mongo = mongo)

@app.route('/help')
def help():
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
   
    cfg['descriptors']['VGG16_false']         = True if request.form.get('VGG16_false')     != None else False
    cfg['descriptors']['XCEPTION_false']      = True if request.form.get('XCEPTION_false')  != None else False
    cfg['descriptors']['XCEPTION_true']       = True if request.form.get('XCEPTION_true')   != None else False
    cfg['descriptors']['MOBILENET_false']     = True if request.form.get('MOBILENET_false') != None else False
    
    cfg['descriptors']['VGG16_false_pca']         = True if request.form.get('VGG16_false_pca')     != None else False
    cfg['descriptors']['XCEPTION_false_pca']      = True if request.form.get('XCEPTION_false_pca')  != None else False
    cfg['descriptors']['XCEPTION_true_pca']       = True if request.form.get('XCEPTION_true_pca')   != None else False
    cfg['descriptors']['MOBILENET_false_pca']     = True if request.form.get('MOBILENET_false_pca') != None else False
    
    cfg['descriptors']['VGG16_false_rmac']         = True if request.form.get('VGG16_false_rmac')     != None else False
    cfg['descriptors']['XCEPTION_false_rmac']      = True if request.form.get('XCEPTION_false_rmac')  != None else False
    cfg['descriptors']['XCEPTION_true_rmac']       = True if request.form.get('XCEPTION_true_rmac')   != None else False
    cfg['descriptors']['MOBILENET_false_rmac']     = True if request.form.get('MOBILENET_false_rmac') != None else False

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
        path = 'static/img_loaded/' + file.filename
        cfg['input']['img_path'] = path
        cfg['input']['is_in_database'] = True if file.filename in os.listdir('static/db/') else False


    # if file is not allowed, flash error
    else:
        flash('[FILE] Wrong file format')
  
if __name__ == '__main__':
    app.run(debug=True)