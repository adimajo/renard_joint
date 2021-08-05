"""
API module
Creating classes to learn a model, predict and
serve these predictions.

.. autosummary::
    :toctree:

    models
    train
    predict
    wsgi
    gini
"""

from flask.json import jsonify
from flask_restful import reqparse, Resource
from loguru import logger

from renard_joint._scripts import spert
from renard_joint.spert import SpertConfig

str_required = {
    'type': str,
    'required': True,
    'default': None
}


int_required = {
    'type': int,
    'required': True,
    'default': None
}


train_parser = reqparse.RequestParser()
train_parser.add_argument('dataset',
                          help="The name of the dataset to train on, one of 'conll04', 'scierc', 'internal'; for now, "
                               "hyperparameters not available through API.",
                          **str_required)

evaluate_parser = reqparse.RequestParser()
evaluate_parser.add_argument('dataset',
                             help="The name of the dataset to compute the performance, one of 'conll04', 'scierc', 'internal'.",
                             **str_required)

predict_parser = reqparse.RequestParser()
predict_parser.add_argument('dataset',
                            help="The name of the dataset (associated to a model) to use for prediction.",
                            **str_required)
predict_parser.add_argument('checkpoint',
                            help="The checkpoint of the model to use for prediction.",
                            **int_required)
predict_parser.add_argument('sentence',
                            help='The sentence to predict entities and relations.',
                            **str_required)


class Predictor(Resource):
    """
    Flask resource to predict
    """
    def post(self):
        """
        post method for Predictor resource: gets the new predictors in the request, predicts and outputs the score.
        ---
        parameters:
          - in: body
            name: body
            schema:
              id: Predict
              required:
                - dataset
                - checkpoint
                - sentence
              properties:
                dataset:
                  type: string
                  description: The name of the dataset (associated to a model) to use for prediction.
                checkpoint:
                  type: number
                  description: The checkpoint of the model to use for prediction.
                sentence:
                  type: string
                  description: The sentence to predict entities and relations.
        responses:
            200:
                description: output of the model
            400:
                description: model found but failed
            500:
                description: all other server errors
        """
        kwargs = predict_parser.parse_args(strict=True)
        logger.info("Successfully parsed arguments")

        spert_config = SpertConfig(kwargs["dataset"])
        constants, input_generator = spert_config.constants, spert_config.input_generator

        entity_label_map, \
            entity_classes, \
            relation_label_map, \
            relation_classes, \
            tokenizer, \
            relation_possibility = spert.data_prep(constants, input_generator, kwargs["dataset"])

        spert_model = spert.load_model(relation_possibility, constants, kwargs["checkpoint"])


        result = spert.predict(entity_label_map,
                               relation_label_map,
                               tokenizer,
                               constants,
                               input_generator,
                               spert_model,
                               [kwargs["sentence"]])

        logger.info("Successfully predicted")

        try:
            response = jsonify(result)
            response.status_code = 200
        except:
            response = jsonify("Model failed")
            response.status_code = 400

        return response


class Evaluator(Resource):
    """
    Flask resource to train
    """
    def post(self):
        """
        post method for Evaluator resource: will return performance metrics.
        ---
        parameters:
          - in: body
            name: body
            schema:
              id: Evaluate
              required:
                - dataset
              properties:
                dataset:
                  type: string
                  description: The name of the dataset to compute the performance, one of 'conll04', 'scierc', 'internal'.
        responses:
            200:
                description: classification metrics.
        """
        kwargs = train_parser.parse_args(strict=True)
        logger.info("Successfully parsed arguments")

        spert_config = SpertConfig(kwargs["dataset"])
        constants, input_generator = spert_config.constants, spert_config.input_generator

        entity_label_map, \
            entity_classes, \
            relation_label_map, \
            relation_classes, \
            tokenizer, \
            relation_possibility = spert.data_prep(constants, input_generator, kwargs["dataset"])

        spert_model = spert.load_model(relation_possibility, constants, kwargs["checkpoint"])

        result = spert.evaluate(entity_label_map,
                                entity_classes,
                                relation_label_map,
                                relation_classes,
                                constants,
                                input_generator,
                                spert_model,
                                constants.test_dataset)

        try:
            response = jsonify(result)
            response.status_code = 200
        except:
            response = jsonify("Model failed")
            response.status_code = 400

        return response


class Trainer(Resource):
    """
    Flask resource to train
    """
    def post(self):
        """
        post method for Trainer resource: will train a new model and store it.
        ---
        parameters:
          - in: body
            name: body
            schema:
              id: Train
              required:
                - dataset
              properties:
                dataset:
                  type: string
                  description: The name of the dataset to train on, one of 'conll04', 'scierc', 'internal'; for now,
                               hyperparameters not available through API.
        responses:
            200:
                description: model has been trained and saved, outputs some metrics
        """
        kwargs = train_parser.parse_args(strict=True)
        logger.info("Successfully parsed arguments")

        spert_config = SpertConfig(kwargs["dataset"])
        constants, input_generator = spert_config.constants, spert_config.input_generator

        entity_label_map, \
            entity_classes, \
            relation_label_map, \
            relation_classes, \
            tokenizer, \
            relation_possibility = spert.data_prep(constants, input_generator, kwargs["dataset"])

        try:
            spert.train(entity_label_map,
                        entity_classes,
                        relation_label_map,
                        relation_classes,
                        relation_possibility,
                        constants,
                        input_generator)

            response = jsonify("Model saved")
            response.status_code = 200
        except:
            response = jsonify("Model failed")
            response.status_code = 400

        return response
