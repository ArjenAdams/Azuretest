import pickle

from backend.app.application.services.iPredictionService import \
    IPredictionService
from backend.app.infrastructure.predictionRepository import \
    IPredictionRepository
from backend.app.presentation.dto.predictionDTO import PredictionDTO


def load_model():
    loaded_model = pickle.load(open('backend/app/infrastructure/model.plk', 'rb'))
    return loaded_model

class PredictionService(IPredictionService):
    def __init__(self):
        self.model = load_model()

    def get_dataset(self):
        return self.desire_repository.get_data()

    def generate_prediction(self, features):

        model = load_model()
        # ToDo : Maybe pre process the features step ? @Hussin
        prediction = model.predict(features)
        probs = self.model.predict_proba(features)
        certainty_score = probs[0][prediction[0]] * 100  # ToDo : Test if this is correct @Hussin
        print(certainty_score, "certainty score")

        # ToDo Save prediction to database @Thomas
        return prediction, certainty_score

    def get_predictions(self):
        # ToDo Get all predictions from database @Thomas
        repository = IPredictionRepository()
        result = repository.get_all_entities()
        repository.close()

        list_of_predictions = []

        for prediction in result:
            if prediction is None:
                return []
            else:
                list_of_predictions.append(PredictionDTO(prediction=prediction, prediction_id=prediction.id).to_dict())

        return list_of_predictions
