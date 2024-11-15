# from backend.app.application.services.testService import TestService
# from flask import Flask, request, jsonify
# from flask_restx import Api, Resource, fields
# import pandas as pd
#
# api = Api(app, version='1.0', title='Lectoraat', description='Lectoraat API')
# test_service = TestService()
#
#
# class TestController:
#     def __init__(self):
#         self.test_service = TestService()
#
#     @api.route('/prediction_values')
#     class ToBeUsedValues(Resource):
#         @api.doc(description="Get unique values for categorical columns")
#         def get(self):
#             return test_service.to_be_used_values()
#
#     @api.route('/data')
#     class DataResource(Resource):
#         @api.doc(params={
#             'limit': {'description': 'The limit for the number of results - HIGHER THAN 50!', 'type': 'int',
#                       'default': 1000},
#         })
#         def get(self):
#             result = test_service.data_source_get(request.args)
#             return jsonify(result)
#
#         @api.doc(params={
#             'limit': {'description': 'The limit for the number of results - HIGHER THAN 50!', 'type': 'int',
#                       'default': 50},
#             'filter_param': {'description': 'Filter parameter for the query', 'type': 'string'},
#             'filter_value': {'description': 'Filter value for the query', 'type': 'string'}})
#         def post(self):
#             result = test_service.data_source_post(request.args)
#             return jsonify(result)
#
#     @api.route('/similarity')
#     class GetSimilarity(Resource):
#         @api.doc(params={'target_index': 'Target index',
#                          'method': 'jaccard',
#                          'sort': 'Sort',
#                          'n_items': 'Number of items',
#                          'unique_only': 'True',
#                          'exclude_self': 'True'})
#         def get(self):
#             result = test_service.check_similarity_via_get(request.args)
#             return jsonify(result)
#
#         @api.doc(params={'target_index': 'Target index',
#                          'method': 'Method',
#                          'sort': 'Sort',
#                          'n_items': 'Number of items',
#                          'unique_only': 'Unique only',
#                          'exclude_self': 'Exclude self'})
#         def post(self):
#             result = test_service.check_similarity_via_post(request.form)
#             return jsonify(result)
#
#     @api.route('/predict')
#     class Predict(Resource):
#         def get_categorical_values(self):
#             df = test_service.get_data_pd()  # Replace with your own function to retrieve the DataFrame
#             categorical_cols = ['herkomst', 'marital_status', 'diagnose_code', 'diagnose_versie', 'opname_type',
#                                 'medicijn',
#                                 'geslacht']
#             values = {}
#
#             for col in categorical_cols:
#                 unique_values = df[col].unique().tolist()
#                 values[col] = unique_values
#
#             return values
#
#         categorical_cols_values = get_categorical_values()  # function that returns dictionary with lists of unique values for each column
#
#         input_model = api.model('Input', {
#             'herkomst': fields.String(required=True, description='Herkomst', enum=categorical_cols_values['herkomst']),
#             'marital_status': fields.String(required=True, description='Marital Status',
#                                             enum=categorical_cols_values['marital_status']),
#             'diagnose_code': fields.String(required=True, description='Diagnose Code',
#                                            enum=categorical_cols_values['diagnose_code']),
#             'diagnose_versie': fields.String(required=True, description='Diagnose Versie',
#                                              enum=categorical_cols_values['diagnose_versie']),
#             'opname_type': fields.String(required=True, description='Opname Type',
#                                          enum=categorical_cols_values['opname_type']),
#             'medicijn': fields.String(required=True, description='Medicijn', enum=categorical_cols_values['medicijn']),
#             'geslacht': fields.String(required=True, description='Geslacht', enum=categorical_cols_values['geslacht']),
#             'leeftijd': fields.Float(required=True, description='Leeftijd'),
#             'duur_ok': fields.Float(required=True, description='Duur OK'),
#         })
#
#         @api.expect(input_model)
#         @api.doc(
#             description="This functionality gives information about what to fill in to test API",
#             responses={
#                 200: 'Successful prediction',
#                 400: 'Validation Error'
#             })
#         def post(self):
#             # Get form data
#             data = request.get_json()  # getting data in json format
#             print(data, "data")
#
#             # Convert the data to a pandas DataFrame
#             df = pd.DataFrame([data])
#
#             return jsonify(test_service.make_prediction(df))
