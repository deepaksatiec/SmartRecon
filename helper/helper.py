import pandas as pd
import os


def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file(filename, UPLOAD_FOLDER):

    file_format = filename.rsplit('.', 1)[1].lower()
    path = os.path.join(UPLOAD_FOLDER, filename)

    if file_format == 'csv':
        file = pd.read_csv(path, engine='python', sep=None)

    elif file_format == 'xlsx':
        file = pd.read_excel(path)

    elif file_format == 'txt':
        file = pd.read_table(path)

    return file
