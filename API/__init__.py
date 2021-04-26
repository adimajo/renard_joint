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
import numpy as np
from datetime import datetime
from flask_restful import reqparse, Resource
from flask import jsonify
from loguru import logger
import importlib

str_required = {
    'type': str,
    'required': True,
    'default': None
}

datetime_required = {
    'type': lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M'),
    'required': True,
    'default': None
}

int_required = {
    'type': int,
    'required': True,
    'default': None
}

float_required = {
    'type': float,
    'required': True,
    'default': None
}

bool_required = {
    'type': bool,
    'required': True,
    'default': None
}


def check_between_0_1(x):
    """
    Checks if input is between 0 and 1.
    :param x: input
    :type x: float or int
    :return: x
    """
    if not 0 <= x <= 1:
        logger.error("Invalid value for proportion_for_test: must be between 0 and 1.")
    else:
        return x


b0_1_required = {
    'type': lambda x: check_between_0_1(x),
    'required': True,
    'default': None
}

b0_1_not_required = {
    'type': lambda x: check_between_0_1(x),
    'required': False,
    'default': None
}


def check_learner(learner):
    """
    Checks if the weak learner argument can be imported.
    :param learner: weak learner to use in classification
    :type learner: str
    :return: learner
    :rtype: str
    """
    if learner == "ASSEMBLE":
        logger.info("ASSEMBLE method chosen.")
        return learner
    try:
        getattr(importlib.import_module("sklearn.linear_model"), learner)
        logger.info("Specified learner could be imported.")
    except ImportError as e:
        logger.error("Specified learner is not installed / cannot be imported. " + str(e))
    return learner


learner_required = {
    'type': lambda x: check_learner(x),
    'required': True,
    'default': None
}


train_parser = reqparse.RequestParser()
train_parser.add_argument('dataset',
                          help="The name of the dataset to train on: conll04, scierc, internal; for now, "
                               "hyperparameters not available through API.",
                          **str_required)

evaluate_parser = reqparse.RequestParser()
evaluate_parser.add_argument('dataset',
                             help="The name of the dataset to compute the performance: conll04, scierc, internal.",
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
                  type: float
                  description: The checkpoint of the model to use for prediction.
                sentence:
                  type: float
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

        # result = predict(**kwargs)
        # logger.info("Successfully predicted")
        #
        # if not result["score"] == np.nan:
        #     response = jsonify(result)
        #     response.status_code = 200
        # else:
        #     response = jsonify("Model failed")
        #     response.status_code = 400

        return response


class Evaluator(Resource):
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
              id: Predict
              required:
                - dataset
              properties:
                dataset:
                  type: string
                  description: The name of the dataset to compute the performance: conll04, scierc, internal.
        responses:
            200:
                description: classification metrics.
        """
        kwargs = train_parser.parse_args(strict=True)
        logger.info("Successfully parsed arguments")

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
              id: Predict
              required:
                - dataset
              properties:
                dataset:
                  type: string
                  description: The name of the dataset to train on: conll04, scierc, internal; for now,
                               hyperparameters not available through API.
        responses:
            200:
                description: model has been trained and saved, outputs some metrics
        """
        kwargs = train_parser.parse_args(strict=True)
        logger.info("Successfully parsed arguments")

        return response
