import base64
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Load the saved model
model = joblib.load('model_rf.pkl')

# Column list to drop
drop_cols = ['patient_id', 'zkh_opn_start', 'start_operatie', 'outtime',
             'discharge_location', 'verzekering', 'overlijdensdatum',
             'ziekenhuis_verval_vlag', 'zkh_ontheem_start']


def preprocess(df):
    df = df.drop(columns=drop_cols)
    return df


def get_unique_values(df):
    categorical_cols = ['herkomst', 'marital_status', 'diagnose_code',
                        'diagnose_versie', 'opname_type', 'medicijn', 'geslacht']
    values = {col: df[col].unique().tolist() for col in categorical_cols}
    return values


def make_prediction(data):
    df = pd.DataFrame([data])
    df = preprocess(df)
    prediction = model.predict(df)
    confidence = max(model.predict_proba(df)[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    shap.summary_plot(shap_values, df)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot = base64.b64encode(buf.read()).decode('utf8')

    return prediction, confidence, plot
