import os
import pandas as pd
import time
from flask import Flask, render_template, request, flash, redirect, url_for, g, session
from helper import allowed_file, read_file, format_duration, render_table, MachineLearningEngine
from werkzeug.utils import secure_filename
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, recall_score, f1_score
from shutil import move, SameFileError
import plotly.offline as offplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import filters

# Constants 
consts = dict(

    ALLOWED_EXTENSIONS=['csv', 'txt', 'xlsx'],
    UPLOAD_FOLDER='static/upload',
    TEMP_FOLDER='static/temp',
    DATA_FOLDER='static/data',
    OUTPUT_FOLDER='static/output',

    PICKLE_FILE='estimator.pkl',
    PARAM_TABLE='param_algo.xlsx',
    MODEL_OUTPUT='model_output.csv',

    CM_NORMALIZE=True
)

app = Flask(__name__)

# register constants
app.config.update(consts)

app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'super secret key'

# register the filters
app.register_blueprint(filters.blueprint)


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

        if 'Back' in request.form:
            return redirect(url_for('index'))

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            flash('Wrong input format. Can be {}'.format(", ".join(app.config['ALLOWED_EXTENSIONS'])) + ".", 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
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

            table = render_table(g.data.head(10), **{'data-classes': "table table-striped", 'data-show-refresh': "true",
                                                     'data-show-toggle': "true", 'data-show-columns': "true"})

            return render_template('features_selection.html', headers=headers, table=table)
        else:
            return redirect(request.url)

    elif request.method == 'POST':

        if 'Back' in request.form:
            return redirect(url_for('load_data', context=context))

        form_dict = request.form
        session['features'] = defaultdict()

        for key in form_dict:

            if form_dict[key] == 'on':
                feature_name = key.rsplit('_', 1)[0]
                feature_cat = form_dict[feature_name]

                session['features'][feature_name] = feature_cat
                session.modified = True

        if 'target' not in session['features'].values():
            flash('No target selected.', 'error')
            return redirect(request.url)

        if list(session['features'].values()).count('target') > 1:
            flash('Choose one and only one target column.', 'error')
            return redirect(request.url)

        non_target_feat = set(session['features'].values()) - set(['target'])

        if not non_target_feat:
            flash('Choose at least one non target feature.', 'error')
            return redirect(request.url)

        return redirect(url_for('choose_method', context=context))


@app.route('/<string:context>/choose_method', methods=['GET', 'POST'])
def choose_method(context):
    # train model

    if request.method == 'GET':

        return render_template('choose_model.html')

    elif request.method == 'POST':

        if 'Back' in request.form:
            return redirect(url_for('features_selection', context=context))

        if 'filename' in session:
            g.data = read_file(session['filename'], app.config['UPLOAD_FOLDER'])
        else:
            flash('Upload a file and choose features first.', 'error')
            return redirect(request.url)

        features_list = [key for key in session['features'].keys() if session['features'][key] != 'target']
        target = [key for key in session['features'].keys() if session['features'][key] == 'target'][0]

        ml_engine = MachineLearningEngine(X=g.data[features_list], y=g.data[target], feat_types=session['features'])

        # best model
        best_estimator = ml_engine.pipeline(algo=request.form['model_select'])

        # fit the best estimator on the entire data_set(for saving later)
        start = time.time()
        fitted_estimator = ml_engine.fit_estimator(best_estimator)
        end = time.time()

        session['exec_time'] = format_duration(end - start)
        print(session['exec_time'])

        file_path = os.path.join(app.config['TEMP_FOLDER'], app.config['PICKLE_FILE'])
        joblib.dump({'estimator': fitted_estimator,
                     'feat_types': session['features'],
                     'features': features_list,
                     'target': target}, file_path)

        # classic cv score
        session['cv_score'], session['confusion_matrix'] = ml_engine.cv_score(best_estimator,
                                                                              normalize=app.config['CM_NORMALIZE'])

        # cross validation score
        session['k_folds_score'] = ml_engine.k_folds_score(estimator=best_estimator, n_splits=5)

        return redirect(url_for('save_model', context=context))


@app.route('/<string:context>/save_model', methods=['GET', 'POST'])
def save_model(context):
    if request.method == 'GET':

        # confusion matrix
        z = session['confusion_matrix']
        x = ['Action', 'No Action']
        y = ['Action', 'No Action']
        colorscale = 'Greys'

        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z, colorscale=colorscale)
        fig.layout.xaxis.title = 'Predicted'
        fig.layout.yaxis.title = 'Actual'

        cm_chart = offplot.plot(fig, show_link=False, output_type="div", include_plotlyjs=True)

        # estimator param
        pipe = joblib.load(os.path.join(app.config['TEMP_FOLDER'], app.config['PICKLE_FILE']))['estimator']
        params = pipe.steps[-1][-1].get_params()

        param_df = pd.DataFrame({'Parameters': list(params.keys()), 'Value': list(params.values())})
        param_df.to_excel(os.path.join(app.config['TEMP_FOLDER'], app.config['PARAM_TABLE']), index=False)

        html_table = render_table(param_df)

        return render_template('save_model.html', cm=cm_chart, param_table=html_table)
    elif request.method == 'POST':

        if 'Back' in request.form:
            return redirect(url_for('choose_method', context=context))

        else:
            try:
                move(os.path.join(app.config['TEMP_FOLDER'], app.config['PICKLE_FILE']),
                     os.path.join(app.config['DATA_FOLDER'], app.config['PICKLE_FILE']))
            except SameFileError:
                pass
            flash('Model saved', 'success')
        return redirect(request.url)


@app.route('/<string:context>/previous_algo', methods=['GET', 'POST'])
def previous_algo(context):
    if request.method == 'GET':
        return render_template('previous_algo.html')

    elif request.method == 'POST':

        if 'Back' in request.form:
            return redirect(url_for('load_data', context=context))

        if 'metrics' in session:
            session.pop('metrics')

        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['TEMP_FOLDER'], filename))

        try:
            lst_estimator_dic = joblib.load(os.path.join(app.config['TEMP_FOLDER'], filename))

        except KeyError:
            flash('Wrong file format', 'error')
            return redirect(request.url)

        estimator = lst_estimator_dic['estimator']
        features = lst_estimator_dic['features']
        target = lst_estimator_dic['target']
        feat_types = lst_estimator_dic['feat_types']

        data = read_file(session['filename'], app.config['UPLOAD_FOLDER'])

        ml_engine = MachineLearningEngine(X=data[features], y=data[target], feat_types=feat_types)
        y_pred = pd.DataFrame(ml_engine.predict(estimator), columns=[target + ' predicted'])

        output_data = pd.concat([data, y_pred], axis=1)
        output_data.to_csv(os.path.join(app.config['OUTPUT_FOLDER'], app.config['MODEL_OUTPUT']))

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


@app.route('/<string:context>/download', methods=['GET'])
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


if __name__ == '__main__':
    # rmtree(app.config['TEMP_FOLDER'], ignore_errors=True)
    app.run(debug=True)
