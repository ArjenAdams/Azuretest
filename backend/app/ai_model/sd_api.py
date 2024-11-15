import json

from flask import Flask, request
from flask_restx import Resource

from .TypesOfVisualisationMethod import OtherVisualisation
from .CounterFactualType import CounterFactualType
from .ai_api import AI_API

app = Flask(__name__, template_folder="templates")

AI_api = AI_API()


class SD_API(Resource):
    def dispatch_request(self):

        """
        Superclass voor de API van de AI-model. Hierin worden de routing
        van verschillende functies verwerkt.

        Returns:
            Any: De response van de API.
        """
        if request.method == 'GET':
            if request.path == '/sd_api/loadModel':
                return self.load_model()
            elif request.path == '/sd_api/get_similarity':
                return self.get_similarity()
            elif request.path == '/sd_api/getData':
                return self.get_data()
            else:
                return self.get()
        elif request.method == "POST":
            if request.path == "/sd_api/predict":
                return self.predict()
            elif request.path == "/sd_api/get_similarity":
                return self.get_similarity()
            elif request.path == '/sd_api/generate_explanations':
                return self.generate_explanations()
            elif request.path == "/sd_api/counterfactuals":
                return self.counterfactuals()
            elif request.path == "sd_api/other_visualization":
                return self.other_visualization()
            else:
                return self.post()
        else:
            return super(SD_API, self).dispatch_request()

    def get_patients_with_search_values(self, search_values):
        """
        Is used by the API to retrieve the patients based on the search values.

        Parameters:
        ----------
        search_values: the values to search for.

        Returns:
        -------
        Dataframe: is the dataframe containing the found patients.
        """
        return AI_api.getPatientsWithSearchvalues(search_values)

    def get_data(self):
        """
        Is used by the API to retrieve all patients.

        Returns:
        -------
        DataFrame: is the dataframe containing the fetched patients.

        """
        return AI_api.getData()

    def get(self):
        print("GET request received")
        return {"message": "GET request received"}

    def post(self):
        print("POST request received")
        return {"message": "POST request received"}

    def get_similarity(self, patient_id=None, amount=None, filterData=None):
        """
        Is used by the API to retrieve the similar patients based on the current patient.

        Returns
        -------
        object: dict
        """
        form_id = request.form.get('ID')
        form_amount = request.form.get('amount')
        filter = request.form.get('filter')
        if form_id is not None and form_amount is not None:
            patient_id = form_id
            amount = int(form_amount)
            filterData = filter


        if (request.method == "POST") :
            print("POST request received")

            print(patient_id, "patientID")
            similar_patients = self.get_similar_patients(patient_id, amount, filterData)
            print(similar_patients, "similar patients")
            return similar_patients

    def load_model(self, data=None) -> bool:
        """
        Is used by the API to load another AI-model.

        Parameters:
        ----------
        data: AI-model path

        Returns:
        -------
        bool: is the model successfully loaded
        """
        form_data = request.form.get('path_model_file')
        if form_data is not None:
            data = form_data
        return AI_api.load_model(data)

    # Predict maakt een voorspelling van het (ingeladen) AI-model,
    # op basis van de meegegeven data die uit de frontend komt.
    def predict(self, data=None):
        """
        Is used by the API to retrieve the prediction based on the given
        patient data.

        Parameters:
        ----------
        data: Currently loaded patient to predict.

        Returns:
        -------
        object: the prediction.
        """
        form_data = request.form.get('data')
        if form_data is not None:
            # Get form data
            data = [json.loads(form_data)]

        print(data, "data")

        # Generates the prediction based on input data.
        prediction = self.generate_prediction(data)
        print(prediction, "prediction")

        # Calculate the confidence score for the prediction and input data
        # in percents
        confidence = self.calculate_probability(data)
        print(confidence, "confidence")

        response = {
            'prediction': float(prediction),
            'confidence': float(confidence),
        }
        return json.dumps(response)

    def counterfactuals(self, data=None, counterfactual_type=None, amount=None):
        try:
            form_counterfactual_type = request.form.get('method')
            form_data = request.form.get('data')
            form_amount = request.form.get('amount')
            if form_counterfactual_type is not None and form_data is not None:
                counterfactual_type = form_counterfactual_type
                data = [json.loads(form_data)]
                amount = int(form_amount)

            # Check if the value is a valid enum member
            if counterfactual_type in CounterFactualType.__members__:
                # Use the value
                print("Valid counterfactual type:", counterfactual_type)

                # Generates the counterfactuals based on input data and the
                # method.
                counterfactuals = self.generate_counterfactuals(
                    data,
                    counterfactual_type,
                    amount
                )
                print(counterfactuals, "counterfactuals")
                if counterfactuals is None:
                    raise ValueError("No counterfactuals found!")
                return counterfactuals
            else:
                raise ValueError(
                    "Invalid counterfactual type: " + counterfactual_type
                )

        except ValueError as e:
            print(e)

    def generate_explanations(self, data=None):
        """
        Is used by the API to generate and retrieve explanations based on
        the input data.

        Parameters:
        ----------
        data: form input data.

        Returns:
        -------
        dict: an dict with the keys: type (image or html) and content
        (the base64 for image and html for html).
        """

        form_data = request.form.get("data")
        if form_data is not None:
            # Get form data
            data = [json.loads(form_data)]

        print(data, "data")
        print(data["type_visualization"])

        if data["type_visualization"] in OtherVisualisation.__members__.keys():
            return AI_api.other_visualization(
                type_visualization=data["type_visualization"],
                kwargs=[
                    data["get_train"],
                    data["CorrMatrixMethod"],
                    data["ImportanceType"],
                    data["max_display"],
                ],
            )
        else:
            return AI_api.get_shap_visualization(
                data["type_visualization"],
                data["range"],
                data["feature_name"],
                data["interaction_index"],
                data["index"],
                data["max_display"],
                data["list_indices"],
            )

    def other_visualization(self, data=None):

        form_data = request.form.get("data")
        if form_data is not None:
            # Get form data
            data = [json.loads(form_data)]

        print(data, "data")
        print(data["type_visualization"])
        return AI_api.other_visualization(data["type_visualization"], data["get_train"])

    # Generate Probability generates the prediction based on the input data.
    def generate_prediction(self, data):
        """
        Is used by the predict method to generate a prediction based on the
        input patient.

        Parameters:
        ----------
        data: input patient.

        Returns:
        -------
        float: prediction
        """
        # Uses the AI-api to generate a prediction.
        prediction = AI_api.predict(data)
        return prediction

    # Calculate Probability calculates the confidence (zekerheidsscore) from
    # the prediction and input data in percents.
    def calculate_probability(self, data):
        """
        Is used by the predict method to generate a probability based on the
        input patient.

        Parameters:
        ----------
        data: input patient.

        Returns:
        -------
        float: calculated probability score.
        """
        # Uses the AI-api to calculate the confidence (zekerheidsscore).
        score = AI_api.predict_probability(data)
        if score < 0.5:
            score = 1 - score
        print("Certainty score: " + str(round(score * 100, 2)) + "%")

        return round(score * 100, 2)

    # Generate Counterfactuals generates the counterfactuals based on the
    # input data and the prediction.
    def generate_counterfactuals(
            self, data, method,
            hoeveelheid_counterfactuals
    ):
        """
        Is used by the counterfactuals method to generate counterfactuals based on the
        input values.

        Parameters:
        ----------
        data: input patient.

        method: counterfactuals method.

        hoeveelheid_counterfactuals: aantal counterfactuals.

        Returns:
        -------
        Dataframe: dataframe with generated counterfactuals.
        """
        # Uses the AI-api to generate counterfactuals.
        counterfactuals = AI_api.get_counterfactual(data, CounterFactualType[method.upper()],
                                                    hoeveelheid_counterfactuals)
        return counterfactuals

    def get_similar_patients(self, patient_id, hoeveelheid, filterData):
        """
        Is used by the get_similarity method to fetch similar patients based
        on the current patient id and the expected amount of similar patients.

        Parameters:
        ----------
        patient_id: the patient id.

        hoeveelheid: expected amount of similar patients.

        Returns:
        -------
        dict: dictionary with similar patients split in a
        list with similar patient data objects, a list with similar patient
        prediction scores and a list with similar patient confidence scores.
        """
        similar_patients = AI_api.get_similarity_maar_dan_beter(patient_id, hoeveelheid, filterData)
        return similar_patients


if __name__ == "__main__":
    app.run(debug=True)
