import json
from pathlib import Path

import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse


from .ai_model.sd_api import SD_API


def create_app():
    app = Flask(__name__)

    # Enable CORS for all domains
    CORS(app)

    sd_api = SD_API()

    api = Api(
        app, version='1.0', title='Lectoraat',
        description='Lectoraat API'
    )
    # # api.add_resource(TestController, '/test')
    api.add_resource(SD_API, '/sd_api/loadModel')
    api.add_resource(SD_API, '/sd_api')
    api.add_resource(SD_API, '/sd_api/getData')
    api.add_resource(SD_API, '/sd_api/get_similarity')
    api.add_resource(SD_API, '/sd_api/predict')
    api.add_resource(SD_API, '/sd_api/generate_explanations')
    api.add_resource(SD_API, '/sd_api/counterfactuals')

    def get_categorical_values():
        mod_path = Path(__file__).parent
        df = pd.read_stata(
            (mod_path / 'ai_model/Modellen/MIMIC-IV.dta').resolve()
        )
        categorical_cols = ['age', 'weight', 'gender', 'temperature',
                            'heart_rate', 'resp_rate',
                            'spo2', 'sbp', 'dbp', 'mbp', 'wbc', 'hemoglobin',
                            'platelet', 'bun', 'cr', 'glu',
                            'Na', 'Cl', 'K', 'Mg', 'Ca', 'P', 'inr', 'pt',
                            'ptt', 'bicarbonate', 'aniongap', 'gcs', 'vent',
                            'crrt', 'vaso', 'seda', 'sofa_score', 'ami', 'ckd',
                            'copd', 'hyperte', 'dm', 'aki', 'stroke',
                            'AISAN', 'BLACK', 'HISPANIC', 'OTHER', 'WHITE',
                            'unknown', 'CCU', 'CVICU', 'MICU', 'MICU/SICU',
                            'NICU', 'SICU', 'TSICU'
                            ]
        values = {}

        for col in categorical_cols:
            unique_values = df.columns.unique().tolist()
            values[col] = unique_values

        return values

    # function that returns dictionary with lists of unique values for each
    # column
    categorical_cols_values = get_categorical_values()

    input_new_model = api.model(
        'Input', {
            'age': fields.Integer(
                required=True, description='Leeftijd in jaren',
                enum=categorical_cols_values['age']
            ),
            'weight': fields.Float(
                required=True, description='Gewicht in kg',
                enum=categorical_cols_values['weight']
            ),
            'gender': fields.Integer(
                required=True, description='Geslacht',
                enum=categorical_cols_values['gender']
            ),
            'temperature': fields.Integer(
                required=True,
                description='Temperatuur in graden celsius',
                enum=categorical_cols_values[
                    'temperature']
            ),
            'heart_rate': fields.Integer(
                required=True,
                description='Hartslag per minuut',
                enum=categorical_cols_values[
                    'heart_rate']
            ),
            'resp_rate': fields.Integer(
                required=True,
                description='Ademhaling snelheid per minuut',
                enum=categorical_cols_values['resp_rate']
            ),
            'spo2': fields.Integer(
                required=True,
                description='Zuurstof gehalte in bloed in procenten',
                enum=categorical_cols_values['spo2']
            ),
            'sbp': fields.Integer(
                required=True,
                description='Systolic blood pressure van de patient in mmHg. '
                            '(Bovenste bloeddruk)',
                enum=categorical_cols_values['sbp']
            ),
            'dbp': fields.Integer(
                required=True,
                description='Diastolic blood pressure van de patient in '
                            'mmHg. (Onderste bloeddruk)',
                enum=categorical_cols_values['dbp']
            ),
            'mbp': fields.Integer(
                required=True,
                description='Mean blood pressure van de patient in mmHg. '
                            '(Gemiddelde bloeddruk van een periode van tijd)',
                enum=categorical_cols_values['mbp']
            ),
            'wbc': fields.Float(
                required=True,
                description='Hoeveelheid witte bloedcellen in 10^3/uL',
                enum=categorical_cols_values['wbc']
            ),
            'hemoglobin': fields.Float(
                required=True,
                description='Hemoglobin gehalte in het bloed van de patient '
                            'in g/dL. (Specifieke eiwit in het bloed dat '
                            'zuurstof transporteert)',
                enum=categorical_cols_values['hemoglobin']
            ),
            'platelet': fields.Integer(
                required=True,
                description='Aantal bloedplaatjes in het bloed van de '
                            'patient in 10^3/uL',
                enum=categorical_cols_values['platelet']
            ),
            'bun': fields.Integer(
                required=True,
                description='Blood area nitrogen gehalte in het bloed van de '
                            'patient in mg/dL. (Indicator van de nierfunctie)',
                enum=categorical_cols_values['bun']
            ),
            'cr': fields.Float(
                required=True,
                description='Creatine gehalte in het bloed van de patient in '
                            'mg/dL.',
                enum=categorical_cols_values['cr']
            ),
            'glu': fields.Integer(
                required=True,
                description='Suiker gehalte in het bloed van de patient in '
                            'mg/dL.',
                enum=categorical_cols_values['glu']
            ),
            'Na': fields.Integer(
                required=True,
                description='Natrium gehalte in het bloed van de patient in '
                            'mmol/L.',
                enum=categorical_cols_values['Na']
            ),
            'Cl': fields.Integer(
                required=True,
                description='Chloride gehalte in het bloed van de patient in '
                            'mmol/L.',
                enum=categorical_cols_values['Cl']
            ),
            'K': fields.Float(
                required=True,
                description='Kalium gehalte in het bloed van de patient in '
                            'mmol/L.',
                enum=categorical_cols_values['K']
            ),
            'Mg': fields.Float(
                required=True,
                description='Magnesium gehalte in het bloed van de patient '
                            'in mmol/L.',
                enum=categorical_cols_values['Mg']
            ),
            'Ca': fields.Float(
                required=True,
                description='Calcium gehalte in het bloed van de patient in '
                            'mmol/L.',
                enum=categorical_cols_values['Ca']
            ),
            'P': fields.Float(
                required=True,
                description='Fosfor gehalte in het bloed van de patient in '
                            'mg/dL.',
                enum=categorical_cols_values['P']
            ),
            'inr': fields.Float(
                required=True,
                description='International normalized ratio van de patient. '
                            '(bloedstolling test) (Hoelang het duurt voordat '
                            'het bloed stolt. lager is beter)',
                enum=categorical_cols_values['inr']
            ),
            'pt': fields.Float(
                required=True,
                description='Prothrombin time van de patient in seconden. '
                            '(bloedstolling test)',
                enum=categorical_cols_values['pt']
            ),
            'ptt': fields.Float(
                required=True,
                description='Partial thromboplastin time van de patient in '
                            'seconden.(bloedstolling test)',
                enum=categorical_cols_values['ptt']
            ),
            'bicarbonate': fields.Integer(
                required=True,
                description='Bicarbonaat gehalte in het bloed van de patient '
                            'in mmol/L.',
                enum=categorical_cols_values[
                    'bicarbonate']
            ),
            'aniongap': fields.Integer(
                required=True,
                description='Anion gap van de patient in mmol/L. (Het '
                            'verschil tussen de gemeten positieve en '
                            'negatieve ionen in het bloed) (huh interessant)',
                enum=categorical_cols_values['aniongap']
            ),
            'gcs': fields.Integer(
                required=True,
                description='Glasgow Coma Scale score van de patient.(schaal '
                            'van 3-15. hoe lager hoe slechter)',
                enum=categorical_cols_values['gcs']
            ),
            'vent': fields.Integer(
                required=True,
                description='of de patient aan de beademing ligt. (Ja/Nee)',
                enum=categorical_cols_values['vent']
            ),
            'crrt': fields.Integer(
                required=True,
                description='of de patient aan de dialyse ligt.(Ja/Nee)',
                enum=categorical_cols_values['crrt']
            ),
            'vaso': fields.Integer(
                required=True,
                description='of de patient aan de vasoconstrictors ligt. '
                            '(Ja/Nee) (Verhoogt de bloeddruk)',
                enum=categorical_cols_values['vaso']
            ),
            'seda': fields.Integer(
                required=True,
                description='of de patient aan de sedatives ligt. (Ja/Nee)',
                enum=categorical_cols_values['seda']
            ),
            'sofa_score': fields.Integer(
                required=True,
                description='Sequential Organ Failure Assessment score van '
                            'de patient. van eerste opname (schaal van 0-24. '
                            'hoe hoger hoe slechter)',
                enum=categorical_cols_values[
                    'sofa_score']
            ),
            'ami': fields.Integer(
                required=True,
                description='of de patient een Acute Myocardial Infarction '
                            'heeft gehad. (Ja/Nee)',
                enum=categorical_cols_values['ami']
            ),
            'ckd': fields.Integer(
                required=True,
                description='of de patient een Chronic Kidney Disease '
                            'heeft. (Ja/Nee)',
                enum=categorical_cols_values['ckd']
            ),
            'copd': fields.Integer(
                required=True,
                description='of de patient een Chronic Obstructive Pulmonary '
                            'Disease heeft.(Ja/Nee)',
                enum=categorical_cols_values['copd']
            ),
            'hyperte': fields.Integer(
                required=True,
                description='of de patient een Hypertension heeft. (Ja/Nee)',
                enum=categorical_cols_values['hyperte']
            ),
            'dm': fields.Integer(
                required=True,
                description='of de patient een Diabetes Mellitus heeft. '
                            '(Ja/Nee)',
                enum=categorical_cols_values['dm']
            ),
            'aki': fields.Integer(
                required=True,
                description='of de patient een Acute Kidney Injury heeft '
                            'gehad. (Ja/Nee)',
                enum=categorical_cols_values['aki']
            ),
            'stroke': fields.Integer(
                required=True,
                description='of de patient een Stroke heeft gehad. (Ja/Nee)',
                enum=categorical_cols_values['stroke']
            ),
            'AISAN': fields.Boolean(
                required=True,
                description='ras van de patient.',
                enum=categorical_cols_values['AISAN']
            ),
            'BLACK': fields.Boolean(
                required=True,
                description='ras van de patient.',
                enum=categorical_cols_values['BLACK']
            ),
            'HISPANIC': fields.Boolean(
                required=True,
                description='ras van de patient.',
                enum=categorical_cols_values['HISPANIC']
            ),
            'OTHER': fields.Boolean(
                required=True,
                description='ras van de patient.',
                enum=categorical_cols_values['OTHER']
            ),
            'WHITE': fields.Boolean(
                required=True,
                description='ras van de patient.',
                enum=categorical_cols_values['WHITE']
            ),
            'unknown': fields.Boolean(
                required=True,
                description='ras van de patient.',
                enum=categorical_cols_values['unknown']
            ),
            'CCU': fields.Boolean(
                required=True,
                description='of de patient op hardbewaking heeft gehad. '
                            '(Ja/Nee)',
                enum=categorical_cols_values['CCU']
            ),
            'CVICU': fields.Boolean(
                required=True,
                description='of de patient een verzorging van de '
                            'Cardiovascular Intensive Care Unit heeft gehad.'
                            ' (Ja/Nee)',
                enum=categorical_cols_values['CVICU']
            ),
            'MICU': fields.Boolean(
                required=True,
                description='of de patient vervoerd is door de mobiele '
                            'intensive care unit. (Ja/Nee)',
                enum=categorical_cols_values['MICU']
            ),
            'MICU/SICU': fields.Boolean(
                required=True,
                description='of de patient een behandeling van de surgical '
                            'intensive care unit heeft gehad tijdens het '
                            'vervoer van de mobiele intensive care unit. '
                            '(Ja/Nee)',
                enum=categorical_cols_values['MICU/SICU']
            ),
            'NICU': fields.Boolean(
                required=True,
                description='of de patient een behandeling van de neonatal '
                            'intensive care unit heeft gehad (is voor '
                            'newborn babies). (Ja/Nee)',
                enum=categorical_cols_values['NICU']
            ),
            'SICU': fields.Boolean(
                required=True,
                description='of de patient een behandeling van de surgical '
                            'intensive care unit heeft gehad. (Ja/Nee)',
                enum=categorical_cols_values['SICU']
            ),
            'TSICU': fields.Boolean(
                required=True,
                description='of de patient een verzorging van de trauma '
                            'intensive care unit heeft gehad. (Ja/Nee)',
                enum=categorical_cols_values['TSICU']
            )
        }
    )

    @api.route('/searchvalues')
    class SearchValues(Resource):
        @api.doc(description="Search based on specific values")
        def post(self):
            parser = reqparse.RequestParser()

            parser.add_argument(
                'ETHNICITY', type=str, required=False,
                store_missing=True
            )
            parser.add_argument(
                'ICU', type=str, required=False,
                store_missing=True
            )
            parser.add_argument(
                'ID', type=int, required=False,
                store_missing=True
            )
            parser.add_argument(
                'AGE', type=int, action='append',
                required=False, store_missing=True
            )
            parser.add_argument(
                'GENDER', type=str, required=False,
                store_missing=True
            )
            parser.add_argument(
                'WEIGHT', type=int, action='append',
                required=False, store_missing=True
            )
            parser.add_argument(
                'SAD', type=bool, required=False,
                store_missing=True
            )
            parser.add_argument(
                'temperature', type=float, required=False,
                default=None
            )
            parser.add_argument(
                'heart_rate', type=int, required=False,
                default=None
            )
            parser.add_argument(
                'resp_rate', type=int, required=False,
                default=None
            )
            parser.add_argument('spo2', type=int, required=False, default=None)
            parser.add_argument('sbp', type=int, required=False, default=None)
            parser.add_argument('dbp', type=int, required=False, default=None)
            parser.add_argument('mbp', type=int, required=False, default=None)
            parser.add_argument(
                'wbc', type=float, required=False,
                default=None
            )
            parser.add_argument(
                'hemoglobin', type=float, required=False,
                default=None
            )
            parser.add_argument(
                'platelet', type=int, required=False,
                default=None
            )
            parser.add_argument('bun', type=int, required=False, default=None)
            parser.add_argument('cr', type=float, required=False, default=None)
            parser.add_argument('glu', type=int, required=False, default=None)
            parser.add_argument('Na', type=int, required=False, default=None)
            parser.add_argument('Cl', type=int, required=False, default=None)
            parser.add_argument('K', type=float, required=False, default=None)
            parser.add_argument('Mg', type=float, required=False, default=None)
            parser.add_argument('Ca', type=float, required=False, default=None)
            parser.add_argument('P', type=float, required=False, default=None)
            parser.add_argument(
                'inr', type=float, required=False,
                default=None
            )
            parser.add_argument('pt', type=float, required=False, default=None)
            parser.add_argument(
                'ptt', type=float, required=False,
                default=None
            )
            parser.add_argument(
                'bicarbonate', type=int, required=False,
                default=None
            )
            parser.add_argument(
                'aniongap', type=int, required=False,
                default=None
            )
            parser.add_argument('gcs', type=int, required=False, default=None)
            parser.add_argument(
                'vent', type=bool, required=False,
                default=None
            )
            parser.add_argument(
                'crrt', type=bool, required=False,
                default=None
            )
            parser.add_argument(
                'vaso', type=bool, required=False,
                default=None
            )
            parser.add_argument(
                'seda', type=bool, required=False,
                default=None
            )
            parser.add_argument(
                'sofa_score', type=int, required=False,
                default=None
            )
            parser.add_argument('ami', type=bool, required=False, default=None)
            parser.add_argument('ckd', type=bool, required=False, default=None)
            parser.add_argument(
                'copd', type=bool, required=False,
                default=None
            )
            parser.add_argument(
                'hyperte', type=bool, required=False,
                default=None
            )
            parser.add_argument('dm', type=bool, required=False, default=None)
            parser.add_argument('aki', type=bool, required=False, default=None)
            parser.add_argument(
                'stroke', type=bool, required=False,
                default=None
            )

            search_values_list = parser.parse_args()

            data = sd_api.get_patients_with_search_values(search_values_list)
            sample_size = min(100, len(data))
            data = data.sample(sample_size)
            data = data.to_json(orient='records')
            return data

    @api.route('/data')
    class DataResource(Resource):
        @api.doc(
            params={
                'limit': {
                    'description': 'The limit for the number of results - '
                                   'HIGHER THAN 50!',
                    'type': 'int',
                    'default': 1000},
            }
        )
        def get(self):
            """
            Gets the patient data from the backend
            Returns
            -------
            object
            """
            data = sd_api.get_data()
            data = data.sample(100)

            data = data.to_json(orient='records')

            return data

    @api.route('/get_similarity')
    class GetSimilarity(Resource):
        @api.expect(input_new_model)
        @api.doc(
            description="This functionality gives information about what to "
                        "fill in to test API",
            responses={
                200: 'Successful prediction',
                400: 'Validation Error'
            }
        )
        def post(self):
            """
            Generates the similar patients based on the input
            Returns
            -------
            object
            """
            # Get form data
            data = request.get_json()  # getting data in json format
            print(data)
            test = sd_api.get_similarity(data['ID'], data['amount'], data['filter'])
            print(test )
            return jsonify(test)

    @api.route('/predict')
    class Predict(Resource):
        @api.expect(input_new_model)
        @api.doc(
            description="This functionality gives information about what to "
                        "fill in to test API",
            responses={
                200: 'Successful prediction',
                400: 'Validation Error'
            }
        )
        def post(self):
            """
            Generates an prediction based on the input
            Returns
            -------
            object
            """
            # Get form data
            data = [request.get_json()]  # getting data in json format
            return jsonify(sd_api.predict(data))

    @api.route('/counterfactual')
    class Counterfactual(Resource):
        @api.expect(input_new_model)
        @api.doc(
            description="This functionality gives information about what to "
                        "fill in to test API",
            responses={
                200: 'Successful prediction',
                400: 'Validation Error'
            }
        )
        def post(self):
            """
            Generates counterfactuals based on the input
            Returns
            -------
            object
            """
            try:
                # Get form data
                data = request.get_json()
                counterfactuals = sd_api.counterfactuals(
                    [data['data']],
                    data['method'],
                    data['amount']
                )

                return json.loads(counterfactuals.to_json(orient='records'))
            except Exception as e:
                print(e)
                return e.__str__(), 400

    @api.route('/generate_explanations')
    class generate_explanations(Resource):
        @api.expect(input_new_model)
        @api.doc(
            description="This functionality gives information about what to "
                        "fill in to test API",
            responses={
                200: 'Successful prediction',
                400: 'Validation Error'
            }
        )
        def post(self):
            """
            Generates the shap explanation graphs based on the input
            Returns
            -------
            object
            """
            # Get form data
            data = request.get_json()

            shap_visualisation = sd_api.generate_explanations(data)

            if shap_visualisation["type"] == "html":
                # For HTML content, directly return the content
                return Response(
                    shap_visualisation["content"],
                    mimetype="text/html"
                )
            elif shap_visualisation["type"] == "image":
                # For image content, return base64 encoded image
                return Response(
                    json.dumps({"image": shap_visualisation["content"]}),
                    mimetype="application/json"
                )
            else:
                return jsonify({"error": "Invalid visualization type."})

    return app
