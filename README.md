In this simple project, we're serving an Image Classification model using `FastAPI`. The API exposes a pre-trained model based on the CIFAR10 dataset and has been built using two approaches

1. Using `pytorch`
2. Using `GluonCV`

Given the large size of the `torch` package, and consequently a longer build time in Docker, I explored using  `GluonCV` and `MXNet`.The `GluonCV` based approach provides better accuracy and smaller build size in Docker.

There is a separate `gluoncv` branch in the repo for the second approach. The major difference is only in the `predict()` method of the classifier.

### Technologies Used

The project uses  `FastAPI`  for model serving. We're using a pre-trained model from `gluoncv-model-zoo`

### Running the app

After installing necessary packages, using `pip install -r requirements.txt` from the root folder, use the following command to run the app from the project root directory-

`uvicorn api.main:app`

In case you want to enable hot-reload for development purposes, add the —reload flag to the above command:

`uvicorn api.main:app --reload`

FastAPI exposes two endpoints: `/healthcheck` and `/predict`

You can either test out the endpoints using Postman or using the built-in documentation explorer which can be accessed by visiting **[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)** from your browser.

### Unit Tests

There are simple tests added to the project which check the endpoints for a valid response. To run the test cases use `pytest` from the project root directory.

The test cases are defined in the `tests/test_main.py` file.

### Run with Docker

The API can also run as a Docker container. To build a docker image for the project, ensure you have docker installed and navigate to the project root. From there, execute the following command:

`docker build -t cifar10-classifier-api .`

After the image is successfully built, run the following commands to run the container.

`docker run -p 8080:8080 cifar10-classifier-api`

You can then access visit [http://0.0.0.0:8080/docs](http://0.0.0.0:8080/docs) from your browser.  From there you can upload an image through `/predict` endpoint and the response will be shown on the page itself.