import base64
from io import BytesIO

import joblib
# import dice_ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from flask import Flask, request, render_template
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder

from backend.app import config

# from dice_ml import Dice
# Flask app
app = Flask(__name__, template_folder='templates')

# Load the saved model
model = joblib.load('model_rf.pkl')

# Column list to drop
drop_cols = ['patient_id', 'zkh_opn_start', 'start_operatie', 'outtime', 'discharge_location', 'verzekering',
             'overlijdensdatum', 'ziekenhuis_verval_vlag', 'zkh_ontheem_start']


def get_data():
    conn = psycopg2.connect(config.getDatabaseURI())
    cur = conn.cursor()
    # Rollback the current transaction
    conn.rollback()
    #print("Connected to database successfully!")
    #print("Executing query...")
    # Your query here
    query = """ SELECT
  p.subject_id AS patient_id,
  p.gender AS geslacht,
  p.anchor_age + EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP)) - p.anchor_year AS leeftijd,
  a.admittime AS zkh_opn_start,
  a.hospital_expire_flag AS ziekenhuis_verval_vlag,
  a.dischtime AS zkh_ontheem_start,
  a.admission_type AS opname_type,
  a.discharge_location, 
  a.insurance AS verzekering,
  a.insurance AS herkomst,
  a.marital_status,
  d.icd_code AS diagnose_code,
  d.icd_version AS diagnose_versie,
  p.dod AS overlijdensdatum,
  tr.intime AS start_operatie,
  tr.outtime,


presc.drug AS medicijn,
EXTRACT(EPOCH FROM (tr.outtime::timestamp - tr.intime::timestamp)) AS duur_ok
FROM
  mimiciv_hosp.patients p
JOIN
  mimiciv_hosp.admissions a
ON
  p.subject_id = a.subject_id
JOIN
  mimiciv_hosp.diagnoses_icd d 
ON
  a.hadm_id = d.hadm_id
JOIN
  mimiciv_hosp.d_icd_diagnoses icd 
ON
  d.icd_code = icd.icd_code AND d.icd_version = icd.icd_version
JOIN
  mimiciv_hosp.transfers tr 
ON
  a.hadm_id = tr.hadm_id
JOIN
  (SELECT * FROM mimiciv_hosp.prescriptions LIMIT 1000) presc 
ON
  a.hadm_id = presc.hadm_id

ORDER BY p.subject_id DESC

LIMIT 1000;"""

    # Execute the SQL query
    cur.execute(query)
    # Fetch all the records
    tuples = cur.fetchall()

    # Get the column names for the DataFrame
    column_names = [desc[0] for desc in cur.description]

    # Create a pandas DataFrame
    df = pd.DataFrame(tuples, columns=column_names)

    # Close the cursor and connection
    cur.close()
    conn.close()
    #print("Query executed successfully!")
    #print(df.head())
    return df


# Function to preprocess new data
def preprocess(df, drop_cols):
    df = df.drop(columns=drop_cols)
    return df


def get_similarity_list(data, target_index,
                        method='jaccard',
                        sort='descending',
                        n_items=-1,
                        unique_only=False,
                        exclude_self=True
                        ):
    one_hot_data = OneHotEncoder(dtype=bool).fit_transform(data).toarray()

    similarity_df = data.copy()
    similarity_df['similarity'] = list(
        1 - pairwise_distances(one_hot_data, Y=np.array(one_hot_data[target_index]).reshape(1, -1), metric=method)[:,
            0])
    similarity_df['n_duplicates'] = data.groupby(list(data.columns), sort=False)[(list(data.columns))[0]].transform(
        'size') - 1
    similarity_df['index'] = similarity_df.index

    if unique_only:
        similarity_df = similarity_df.drop_duplicates(
            subset=list(similarity_df.drop(columns=['index', 'similarity', 'n_duplicates']).columns))

    if exclude_self:
        similarity_df = similarity_df.drop(index=similarity_df.iloc[target_index].name)

    similarity_list = similarity_df[['index', 'similarity', 'n_duplicates']].to_dict('records')

    if sort == 'descending':
        sort_direction = -1
    elif sort == 'ascending':
        sort_direction = 1
    else:
        return similarity_list[:n_items]

    return sorted(similarity_list, key=lambda d: d['similarity'])[::sort_direction][:n_items]


@app.route('/get_similarity', methods=['GET', 'POST'])
def get_similarity():
    df = get_data()
    indices = df.index.tolist()

    if request.method == 'POST':
        target_index = int(request.form.get('target_index'))
        method = request.form.get('method')
        sort = request.form.get('sort')
        n_items = int(request.form.get('n_items'))
        unique_only = request.form.get('unique_only') == 'True'
        exclude_self = request.form.get('exclude_self') == 'True'

        result = get_similarity_list(data=df,
                                     target_index=target_index,
                                     method=method,
                                     sort=sort,
                                     n_items=n_items,
                                     unique_only=unique_only,
                                     exclude_self=exclude_self)
        return render_template('results.html', result=result, indices=indices)

    return render_template('get_similarity.html', indices=indices)


@app.route('/', methods=['GET'])
def home():
    # Get data
    df = get_data()
    categorical_cols = ['herkomst', 'marital_status', 'diagnose_code', 'diagnose_versie', 'opname_type', 'medicijn',
                        'geslacht']

    # Get unique values for categorical columns
    values = {col: df[col].unique().tolist() for col in categorical_cols}

    # Render the form template with the unique values
    return render_template("form.html", values=values)


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    #print(data, "data")

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame([data])

    # Use the model to predict the class and probability
    prediction = model.predict(df)
    #print(prediction, "prediction")
    # calculate the confidence score for the prediction in precents

    prediction_proba = model.predict_proba(df)
    confidence = round(prediction_proba[0][prediction[0]] * 100, 2)
    #print(confidence, "confidence")

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close('all')

    # calculate the confidence score for the prediction in precents

    # # Create an instance of DiCE
    # d = Dice(df, model)
    #
    # # Generate counterfactual examples
    # dice_exp = d.generate_counterfactuals(df, total_CFs=5, desired_class="opposite")
    #
    # # Get the counterfactual examples
    # counterfactuals = dice_exp.cf_examples_list

    # Create response
    response = {
        'prediction': prediction[0],
        'confidence': confidence,
        # 'counterfactuals': counterfactuals
        'plot': f"data:image/png;base64,{data}"
    }

    return render_template('results.html',
                           prediction=prediction[0],
                           confidence=confidence,
                           plot=f"data:image/png;base64,{data}")


if __name__ == '__main__':
    app.run(debug=True)
