import os

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.secret_key = "super secret key"

cfg = {}

@app.route("/", methods = ['GET', 'POST'])
def index():

    # apres une recherche intense et reflechie les img sont stockee dans img_result
    cfg['result'] = []
    for file in os.listdir('static/img_result'):
        cfg['result'].append("static/img_result/" + str(file))

    if request.method == "POST" and "form1" in request.form:
        cfg['sift'] = request.form.get('SIFT')
        cfg['bgr'] = request.form.get('BGR')
        cfg['glcm'] = request.form.get('GLCM')
        cfg['hog'] = request.form.get('HOG')
        cfg['hsv'] = request.form.get('HSV')
        cfg['lbp'] = request.form.get('LBP')
        cfg['orb'] = request.form.get('ORB')
        cfg['dl'] = request.form.get('DL')

        if not (cfg['sift'] or cfg['bgr'] or cfg['glcm'] or cfg['hog'] or cfg['hsv'] or cfg['lbp'] or cfg['orb'] or cfg['dl']):
            flash('Pas de descripteur sélectionné')
            return redirect(request.url)
        
        cfg['distance'] = request.form.get('distance')
        if not cfg['distance']:
            flash('Pas de distance sélectionnée')
            return redirect(request.url)

    if request.method == 'POST' and "form2" in request.form:
        upload_file()
            
    return render_template("index.html", cfg = cfg)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

def upload_file():
    if 'file' not in request.files:
        flash('Pas de fichier')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('Pas de fichier sélectionné')
        return redirect(request.url)
    # vérifier que le format est accepté
    if file.filename != '' and allowed_file(file.filename):
        os.makedirs(os.path.join('static', 'img_loaded'), exist_ok=True)
        file.save(os.path.join('static', 'img_loaded', secure_filename(file.filename)))
        cfg['img_url'] = "static/img_loaded/" + file.filename
    else:
        flash('Erreur: Format de fichier non accepté, veuillez mettre un png/jpg/jpeg')
  
if __name__ == "__main__":
    app.run(debug=True)