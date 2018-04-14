import os
from flask import Flask, render_template, request, flash, redirect, url_for
from helper import allowed_file
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = ['csv', 'txt', 'xlsx']
UPLOAD_FOLDER = '/upload'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'GET':
        return render_template('index.html')

    # Upload file
    elif request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['data']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('home',
                                    #filename=filename
                                    ))


if __name__ == '__main__':
    app.run(debug=True)