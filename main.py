from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from flasgger import Swagger
from flask_cors import CORS
from datetime import datetime
from dateutil import parser
import numpy as np

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

def prepare_data(data):
    features = []
    for item in data:
        actual_job = item.get('actualJob', {})
        passport = item.get('passport', {})
        loans = item.get('loans', [{}])

        if actual_job is None:
            actual_job = {}
        if loans is None:
            loans = []

        feature_vector = [
            actual_job.get('salaryAmount', 0),
            parser.parse(actual_job.get('startDate')).timestamp(),
            parser.parse(passport.get('birthDate')).timestamp(),
            loans[0].get('amount', 0) if loans else 0,
            # parser.parse(loans[0].get('startDate')).timestamp(),
        ]
        features.append(feature_vector)
    return np.array(features)


def perform_clustering(data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_


@app.route('/cluster', methods=['POST'])
def cluster_data():
    """
    Кластеризация данных.

    ---
    parameters:
      - name: data
        in: body
        required: true
        schema:
          type: object
          properties:
            data:
              type: array
              items:
                type: object
                # (описание структуры данных здесь)

    responses:
      200:
        description: Successful operation
        schema:
          type: object
          properties:
            clusters:
              type: array
              items:
                type: integer
              description: Кластеры, к которым отнесены данные
      400:
        description: Invalid input
    """
    input_data = request.get_json()



    prepared_data = prepare_data(input_data)
    clusters = perform_clustering(prepared_data)

    result = {"clusters": clusters.tolist()}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)