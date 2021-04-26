from flask import Flask, jsonify
import flask_restful as restful
from flask_swagger import swagger
from API import Predictor, Evaluator, Trainer
from rsenard_joint import __version__


def create_app():
    """
    Creates the model serving Flask app
    :return: Flask app
    """
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    api = restful.Api(app)
    api.add_resource(Predictor, '/predict')
    api.add_resource(Evaluator, '/evaluate')
    api.add_resource(Trainer, '/train')

    @app.route("/spec")
    def spec():
        swag = swagger(app)
        swag['info']['version'] = __version__
        swag['info']['title'] = "Data Harvesting"
        return jsonify(swag)

    return app


app = create_app()


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=8000)
