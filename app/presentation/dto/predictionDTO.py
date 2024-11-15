class PredictionDTO:
    def __init__(self, prediction, prediction_id):
        self.prediction_id = prediction_id
        self.human_body_features = prediction.human_body_features
        self.certainty_score = prediction.certainty_score
        self.medical_sickness = prediction.medical_sickness
        self.medical_sickness_description = prediction.medical_sickness_description

    def to_dict(self):
        return {
            'prediction_id': self.prediction_id,
            'human_body_features': self.human_body_features.features_to_dict(),
            'certainty_score': self.certainty_score,
            'medical_sickness': self.medical_sickness,
            'medical_sickness_description': self.medical_sickness_description,
        }
