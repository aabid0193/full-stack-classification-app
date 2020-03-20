import os
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask import Flask, session


ALLOWED_EXTENSIONS = set(['csv'])
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = CURRENT_PATH + os.path.sep + 'uploads'
RESOURCES_FOLDER = CURRENT_PATH + os.path.sep + 'resources' + os.path.sep
MODELS_FOLDER = CURRENT_PATH + os.path.sep + 'models' + os.path.sep


def create_app(test_config=None):

    app = Flask(__name__, instance_relative_config=True)

    app.config['JWT_SECRET_KEY'] = '\xfc\x0e\xbet\xc5\xca\x91\xb6\x0fx\xa1\xc9\xa0\x1b\xa2V\xefJb\x1f\x05\xcc\xb0n'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.secret_key = '\xfc\x0e\xbet\xc5\xca\x91\xb6\x0fx\xa1\xc9\xa0\x1b\xa2V\xefJb\x1f\x05\xcc\xb0n'
    jwt = JWTManager(app)

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True
                               )
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # index page
    @app.route('/')
    def indexPage():
        session['ok'] = 'test'
        return 'This is the restapi page.'

    # init route for api
    from api_module import api
    app.register_blueprint(api.bp)

    cors = CORS(app, resources={
                r"/api/*": {"origins": "*"}}, supports_credentials=True)
    return app
