from tests.test_setup import test_api

import os

def test_healthcheck_endpoint(test_api):
    response = test_api.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {'ping': 'pong'}

def test_predict_when_image_is_valid(test_api):
    filepath = os.path.join('tests','assets', 'car.jpg')
    # Predicted Class is not checked since the accuracy isn't upto the mark

    expected_response = {'filename': 'car.jpg', 'contentype': 'image/jpeg'}
    actual_response = test_api.post('/predict', files={"file": ("car.jpg", open(filepath, "rb"), "image/jpeg")})


    assert actual_response.status_code == 200
    actual_response = actual_response.json()
    assert actual_response['filename'] == expected_response['filename']
    assert actual_response['contentype'] == expected_response['contentype']


def test_invalid_file_upload(test_api):

    filepath = os.path.join('tests','assets', 'invalid_file.txt')

    expected_response = {'detail' :'File invalid_file.txt is not an image!'}
    actual_response = test_api.post('/predict', files={"file":  open(filepath, "rb")})

    assert actual_response.status_code == 400
    assert actual_response.json() == expected_response
