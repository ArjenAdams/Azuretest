from flask import Blueprint, request, jsonify

from backend.app.application.services.predictionService import \
    PredictionService as predictionService
from backend.app.domain.entities.humanBodyFeatures import \
    HumanBodyFeatures as humanBodyFeatures
from backend.app.infrastructure.desireRepository import \
    DesireRepository as desireRepo


class PredictionController:
    def __init__(self):
        self.prediction_service = predictionService()
        to_test_data_repo = desireRepo()
        to_test_data_repo.get_data()

    def create_prediction(self):
        data = request.get_json()

        features = humanBodyFeatures(
            body_mass_index=data['body_mass_index'],
            asa_score=data['asa_score'],
            respiratory_rate=data['respiratory_rate'],
            oxygen_saturation=data['oxygen_saturation'],
            heart_rate=data['heart_rate'],
            blood_pressure=data['blood_pressure'],
            body_temperature=data['body_temperature'],
            pain_level=data['pain_level'],
            metabolic_rate=data['metabolic_rate'],
            physical_activity=data['physical_activity']
        )

        prediction_dto = self.prediction_service.generate_prediction(features)

        return jsonify(prediction_dto), 200

    def get_predictions(self):
        predictions = self.prediction_service.get_predictions()
        return jsonify(predictions), 200


predictions_blueprint = Blueprint('predictions_blueprint', __name__)
controller = PredictionController()
predictions_blueprint.add_url_rule('/predictions', 'create_prediction', controller.create_prediction, methods=['POST'])
predictions_blueprint.add_url_rule('/predictions/all', 'get_prediction', controller.get_predictions, methods=['GET'])
