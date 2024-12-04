from flask import Blueprint, jsonify, request

from backend.app.application.services.patientService import PatientService
from backend.app.domain.entities.patient import Patient


class PatientController:        
    def __init__(self):
        self.patient_service = PatientService()

    def get_patient(self, patient_id: int):
        patient = self.patient_service.get_patient(patient_id=patient_id)

        if patient is None:
            return jsonify("Not found"), 404

        return jsonify(patient), 200

    
    def searchvalues(self):
        patient = self.patient_service.get_patients_by_searchvalues()

        if patient is None:
            return jsonify("Not found"), 405

        return jsonify(patient), 200

    def add_patient(self, data=None):
        if data is None:
            data = request.get_json()

        if 'name' not in data or 'birth_date' not in data or 'patient_id' not in data or 'features' not in data:
            return jsonify("Invalid request"), 400

        patient = Patient(name=data['name'], birth_date=data['birth_date'], patient_id=data['patient_id'], features=[])

        patient = self.patient_service.add_patient(patient=patient)
        return jsonify({'id': patient.id}), 201


predictions_blueprint = Blueprint('predictions_blueprint', __name__)
controller = PatientController()
predictions_blueprint.add_url_rule('/patient/<int:patient_id>',
                                   'get_prediction', controller.get_patient,
                                   methods=['GET'])
predictions_blueprint.add_url_rule('/patient/searchvalues', 'searchvalues', controller.searchvalues, methods=['GET'])
