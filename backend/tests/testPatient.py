from flask import Flask
from backend.app.application.services.patientService import PatientService
from backend.app.presentation.controllers.patientController import PatientController
from backend.app.domain.entities.humanBodyFeatures import HumanBodyFeatures
from flask import Flask

from backend.app.application.services.patientService import PatientService
from backend.app.domain.entities.humanBodyFeatures import HumanBodyFeatures
from backend.app.presentation.controllers.patientController import \
    PatientController


def test_add_patient():
    # Create an instance of Flask class for our test
    app = Flask(__name__)

    # Context for the Flask application
    with app.app_context():
        patient_service = PatientService()
        patient_controller = PatientController()

        # Test data
        patient_data = {
            'name': "John Doe",
            'birth_date': "1980-02-29",
            'patient_id': "123",
            'features': []
        }

        # Add patient
        response, status_code = patient_controller.add_patient(patient_data)
        assert status_code == 201

        # Now that we've added the patient and gotten their id, we can add the features
        patient_id = response.get_json()['id']
        features = HumanBodyFeatures(
            patient_id=patient_id,
            body_mass_index="20.3", asa_score=2, respiratory_rate=18,
            oxygen_saturation=98, heart_rate=75, blood_pressure="120",
            body_temperature=98, pain_level=3, metabolic_rate=1700,
            physical_activity=30)

        patient = patient_service.get_patient(patient_id)
        patient.features.append(features)


def test_get_patient():
    app = Flask(__name__)

    with app.app_context():
        patient_controller = PatientController()

        # Assume the patient with id 1 exists
        response = patient_controller.get_patient(patient_id=1)
        assert response.status_code == 200
        assert 'id' in response.get_json()
