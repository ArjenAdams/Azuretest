import json
import re

import joblib
import pandas as pd

from ...customJsonDecoder import DecimalEncoder
from ...infrastructure.desireRepository import DesireRepository
from ...infrastructure.predictionRepository import \
    IPredictionRepository as predictionRepository


class TestService:
    def __init__(self):
        self.prediction_repository = predictionRepository()
        self.desire_repo = DesireRepository()

    def get_data_pd(self):
        return self.desire_repo.get_data_pd()

    def make_prediction(self, df):
        # Load the model
        model = joblib.load("app/model_rf.pkl")

        # Use the model to predict the class and probability
        prediction = model.predict(df)

        # calculate the confidence score for the prediction in percent
        prediction_proba = model.predict_proba(df)
        confidence = round(prediction_proba[0][prediction[0]] * 100, 2)

        # calculate the 5 most important features of the Random Forest Model
        importances = model.named_steps['model'].feature_importances_
        feature_names = model.named_steps['preprocessing'].get_feature_names_out()
        # Extract column names from feature names
        column_names = []
        for feature_name in feature_names:
            match = re.search('onehot__(.*?)_', feature_name)
            if match:
                column_names.append(match.group(1))
            elif feature_name.startswith('remainder_'):
                column_names.append(feature_name.replace('remainder__', ''))
            else:
                column_names.append(feature_name)

        # Create a DataFrame with column names and importances
        df_importances = pd.DataFrame({'column': column_names, 'importance': importances})
        print(df_importances)

        # Aggregate importances by column and only save the 5 most important features
        df_importances = df_importances.groupby('column').agg({'importance': 'sum'}).reset_index().nlargest(5,
                                                                                                            'importance')

        # Turn it into a list with tuples
        important_features = list(df_importances.itertuples(index=False, name=None))

        # Create response
        response = {
            'prediction': int(prediction[0]),  # ensure prediction is an int, not int64
            'confidence': float(confidence),  # ensure confidence is a float, not int64
            'important_features': important_features
        }
        return json.dumps(response, cls=DecimalEncoder)

    def check_similarity_via_get(self, data):
        df = self.desire_repo.get_data_pd()
        target_index = int(data.get('target_index'))
        method = data.get('method') == 'jaccard'
        sort = data.get('sort')
        n_items = int(data.get('n_items') == 1)
        unique_only = data.get('unique_only') == 'True'
        exclude_self = data.get('exclude_self') == 'True'

        result = self.desire_repo.get_similarity_list(data=df,
                                                      target_index=target_index,
                                                      method=method,
                                                      sort=sort,
                                                      n_items=n_items,
                                                      unique_only=unique_only,
                                                      exclude_self=exclude_self)
        return result

    def check_similarity_via_post(self, data):
        df = self.desire_repo.get_data_pd()
        target_index = int(data.get('target_index'))
        method = data.get('method')
        sort = data.get('sort')
        n_items = int(data.get('n_items'))
        unique_only = data.get('unique_only') == 'True'
        exclude_self = data.get('exclude_self') == 'True'

        result = self.desire_repo.get_similarity_list(data=df,
                                                      target_index=target_index,
                                                      method=method,
                                                      sort=sort,
                                                      n_items=n_items,
                                                      unique_only=unique_only,
                                                      exclude_self=exclude_self)
        return result

    def to_be_used_values(self):
        # Get data
        df = self.desire_repo.get_data_pd()
        categorical_cols = ['herkomst', 'marital_status', 'diagnose_code', 'diagnose_versie',
                            'opname_type', 'medicijn', 'geslacht']

        # Get unique values for categorical columns
        values = {col: df[col].unique().tolist() for col in categorical_cols}

        return values

    def data_source_get(self, data):
        limit = int(data.get('limit', 1000))
        data = self.desire_repo.get_data(limit=limit)
        data = json.dumps(data, cls=DecimalEncoder)
        return json.loads(data), 200

    def data_source_post(self, data):
        """Post data to the repository and perform calculations"""
        limit = int(data.get('limit', 1000))
        filter_param = data.get('filter_param')
        filter_value = data.get('filter_value')
        data = self.desire_repo.get_data(limit=limit, filter_param=filter_param, filter_value=filter_value)
        data = json.dumps(data, cls=DecimalEncoder)
        return json.loads(data), 200

    def searchvalues(self, searchValueList):
        #patients = self.desire_repo.get_entities_by_searchvalues(searchValueList)
        patients = self.desire_repo.get_data(limit=1000)

        data = json.dumps(patients, cls=DecimalEncoder)
        return json.loads(data), 200

    def get_counterfactuals(self, data):
        # TODO: get counterfactuals from ai model, this is only dummy data
        return data
