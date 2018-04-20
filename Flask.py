import os
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for, g, session, send_file, send_from_directory
from helper import allowed_file, read_file, MachineLearningEngine
from werkzeug.utils import secure_filename
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, recall_score, f1_score
from shutil import move
import plotly.offline as offplot
import plotly.graph_objs as go

ALLOWED_EXTENSIONS = ['csv', 'txt', 'xlsx']
UPLOAD_FOLDER = './upload'
TEMP_FOLDER = './temp'
DATA_FOLDER = './data'
OUTPUT_FOLDER = './output'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['OUTPUT FOLDER'] = OUTPUT_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'super secret key'


@app.route('/', methods=['POST', 'GET'])
def index():

    session.clear()
    return render_template('index.html')


@app.route('/<context>/load_data', methods=['GET', 'POST'])
def load_data(context):

    if request.method == 'GET':
        return render_template('load_data.html')

    # Upload file
    elif request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            session['filename'] = filename
            session.modified = True

            if context == 'find_model':
                return redirect(url_for('features_selection', context=context))

            elif context == 'run_model':
                return redirect(url_for('previous_algo', context=context))


@app.route('/<string:context>/feature_selection', methods=['GET', 'POST'])
def features_selection(context):

    if request.method == 'GET':

        if 'filename' in session:

            g.data = read_file(session['filename'], app.config['UPLOAD_FOLDER'])
            headers = g.data.columns

            return render_template('features_selection.html', headers=headers, data=g.data.head(10))
        else:
            return redirect(request.url)

    elif request.method == 'POST':

        session['features'] = defaultdict()

        form_dict = request.form

        for key in form_dict:

            if form_dict[key] == 'on':

                feature_name = key.rsplit('_', 1)[0]
                feature_cat = form_dict[feature_name]

                session['features'][feature_name] = feature_cat
                session.modified = True

        return redirect(url_for('choose_method', context=context))


@app.route('/<string:context>/choose_method', methods=['GET', 'POST'])
def choose_method(context):
    # train model

    if request.method == 'GET':

        return render_template('choose_model.html')

    elif request.method == 'POST':

        if 'filename' in session:
            g.data = read_file(session['filename'], app.config['UPLOAD_FOLDER'])
        else:
            flash('Upload a file and choose features first.')
            return redirect(request.url)

        features_list = [key for key in session['features'].keys() if session['features'][key] != 'label']
        target = [key for key in session['features'].keys() if session['features'][key] == 'label'][0]

        ml_engine = MachineLearningEngine(X=g.data[features_list], y=g.data[target])

        # best model
        best_estimator = ml_engine.find_best_estimator()

        # cross validation score
        session['cv_score'] = ml_engine.cross_validation_score(estimator=best_estimator)

        # fit the best estimator (for saving later)
        fitted_estimator = ml_engine.fit_estimator(best_estimator)
        joblib.dump({'estimator': fitted_estimator,
                     'features': features_list,
                     'target': target}, os.path.join(app.config['TEMP_FOLDER'], 'best_estimator.pkl'))

        return redirect(url_for('save_model', context=context))


@app.route('/<string:context>/save_model', methods=['GET', 'POST'])
def save_model(context):

    if request.method == 'GET':

        return render_template('save_model.html', score=session['cv_score'])

    elif request.method == 'POST':
        move(os.path.join(app.config['TEMP_FOLDER'], 'best_estimator.pkl'),
             os.path.join(app.config['DATA_FOLDER'], 'best_estimator.pkl'))

        return redirect(url_for('index'))


@app.route('/<string:context>/previous_algo', methods=['GET', 'POST'])
def previous_algo(context):

    if request.method == 'GET':
        return render_template('previous_algo.html')

    elif request.method == 'POST':

        if 'metrics' in session:
            session.pop('metrics')

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['TEMP_FOLDER'], filename))

        try:
            lst_estimator_dic = joblib.load(os.path.join(app.config['TEMP_FOLDER'], filename))

        except KeyError:
            flash('Wrong file format')
            return redirect(request.url)

        estimator = lst_estimator_dic['estimator']
        features = lst_estimator_dic['features']
        target = lst_estimator_dic['target']

        data = read_file(session['filename'], app.config['UPLOAD_FOLDER'])

        ml_engine = MachineLearningEngine(X=data[features])
        y_pred = pd.DataFrame(ml_engine.predict(estimator), columns=[target + ' predicted'])

        output_data = pd.concat([data, y_pred], axis=1)
        output_data.to_csv(os.path.join(app.config['OUTPUT FOLDER'], 'algo_output.csv'))

        if target in data.columns:
            y_true = data[target]

            session['metrics'] = {
                'accuracy': accuracy_score(y_pred=y_pred, y_true=y_true),
                'precision': accuracy_score(y_pred=y_pred, y_true=y_true),
                'recall': recall_score(y_pred=y_pred, y_true=y_true),
                'f1_score': f1_score(y_pred=y_pred, y_true=y_true)
            }
            session.modified = True

        return redirect(url_for('download', context=context))


@app.route('/<string:context>/download', methods=['GET', 'POST'])
def download(context):

    if request.method == 'GET':

        chart = None

        if 'metrics' in session:
            data = [
                go.Bar(
                    x=list(session['metrics'].keys()),
                    y=list(session['metrics'].values()),
                    marker=dict(
                        color='rgb(55, 83, 109)'
                    )
                )
            ]
            layout = go.Layout(
                title='Statistics'
            )

            fig = go.Figure(
                data=data,
                layout=layout
            )

            chart = offplot.plot(fig, show_link=False, output_type="div", include_plotlyjs=True)

        return render_template('download.html', chart=chart)

    elif request.method == 'POST':

        if 'download_output' in request.form:
            send_from_directory(
                    app.config['OUTPUT FOLDER'], 'algo_output.csv', mimetype='text/csv', as_attachment=True)

            os.startfile(app.config['OUTPUT FOLDER'])

        elif 'download_statistics' in request.form:
            pass

        return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
    session.pop('features', None)
