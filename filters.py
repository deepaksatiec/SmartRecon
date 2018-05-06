import jinja2
import flask


# using the method
@jinja2.contextfilter
def safe_path(path):
    if path[:6] == 'static':

        path = path[6:]

        while path[0] == '/' or path[0] == '\\':
            path = path[1:]
    return path


blueprint = flask.Blueprint('filters', __name__)
blueprint.add_app_template_filter(safe_path)
