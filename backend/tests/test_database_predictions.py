import json
import unittest

from flask_testing import TestCase

from backend.app import create_app
from backend.app.infrastructure.predictionRepository import \
    IPredictionRepository


class TestPredictions(TestCase):

    def create_app(self):
        app = create_app()
        app.config['TESTING'] = True
        return app

    def test_create_prediction(self):
        # Prepare test data
        test_data = {
            "body_mass_index": 22,
            "asa_score": 1,
            "respiratory_rate": 12,
            "oxygen_saturation": 98,
            "heart_rate": 60,
            "blood_pressure": "120/80",
            "body_temperature": 37,
            "pain_level": 2,
            "metabolic_rate": 1500,
            "physical_activity": 1
        }

        # Send POST request with test data
        response = self.client.post('/predictions', data=json.dumps(test_data), content_type='application/json')

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Load the response data as a JSON object
        response_data = json.loads(response.data)

        print(response_data)

        self.assertIn('certainty_score', response_data)
        self.assertIn('medical_sickness', response_data)
        self.assertIn('prediction_id', response_data)

        # Check if the entity has been added to the database
        repository = IPredictionRepository()

        prediction = repository.get_entity(response_data['prediction_id'])
        self.assertIsNotNone(prediction)
        print(prediction)
        repository.close()


if __name__ == '__main__':
    unittest.main()
