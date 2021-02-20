import pytest
import app as flask_app
import pandas as pd
#from app import app as flask_app


# testing the response of page for index function
def test_index():
    response = flask_app.app.test_client().get('/')
    assert response.status_code == 200

def test_about():
    response = flask_app.app.test_client().get('/upload2')
    assert response.status_code == 200


def test_service_1():
    response = flask_app.app.test_client().get('/upload_3')
    assert response.status_code == 200


def test_service_2():
    response = flask_app.app.test_client().get('/features')
    assert response.status_code == 200


def test_service_3():
    response = flask_app.app.test_client().get('/model_explanation')
    assert response.status_code == 200


def test_service_4():
    response = flask_app.app.test_client().get('/data_explanation')
    assert response.status_code == 200







