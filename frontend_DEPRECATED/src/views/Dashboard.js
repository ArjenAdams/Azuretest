import React, {useEffect, useState} from "react";
import '../assets/css/font-sizes.css';
import DashboardInput from "../components/Dashboard/DashboardInput";
import DashboardInfoCard
    from "../components/InformationCards/DashboardInfoCard";
import {
    fetchPatientData,
    fetchPredictionData,
    setPredictionData
} from "../handler/sessionStorageHandler";

function convertToBoolean(boolean) {
    if (boolean === 1) {
        return "Ja!"
    }
    return "Nee!"
}

function Dashboard() {
    const [patient, setPatient] = useState({});
    const [prediction, setPrediction] = useState({});
    const [predictionPresent, setPredictionPresent] = useState(false);

    let setStates = (newPrediction) => {
        setPrediction(newPrediction);
        setPredictionPresent(true);
        setPredictionData(newPrediction);
    }

    useEffect(() => {
        const fetchData = async () => {
            const patientData = fetchPatientData()
            setPatient(patientData)
            const predictionData = await fetchPredictionData();
            setPrediction(predictionData);
        };
        fetchData();
    }, [])
    const cardInfo = {
        "Patient ID": patient?.ID || "",
        "Prediction": prediction.prediction,
        "Probability": prediction.confidence
    };

    let gender = "F"
    // Gender converten naar getallen
    if (patient['gender'] === 1) {
        gender = "M"
    }

    if (patient.ID === undefined) {
        return (
            <>
                <div className="content">
                    <DashboardInfoCard cardInformation={cardInfo}/>
                    <p>Loading patient data...</p>
                </div>
            </>
        );
    }
    // The features shown on the input features screen, features that have a default assignment to them instead of "" are required for a prediction to be made
    const humanBodyFeatures = {
        "PatientID": {
            "value": patient.ID,
            "type": "text",
            "description": "Het ID van de Patient"
        },
        "Gender": {
            "value": gender,
            "type": "text",
            "description": "Geslacht"
        },
        "Age": {
            "value": patient.age,
            "type": "number",
            "description": "Leeftijd bij opname in jaren"
        },
        "Weight": {
            "value": patient.weight.toFixed(2),
            "type": "number",
            "description": "Gewicht in kg"
        },
        "Temperature": {
            "value": patient.temperature,
            "type": "number",
            "description": "Temperatuur in graden celsius"
        },
        "Heart Rate": {
            "value": patient.heart_rate,
            "type": "number",
            "description": "Hartslag per minuut"
        },
        "Respiration Rate": {
            "value": patient.resp_rate,
            "type": "number",
            "description": "Ademhaling snelheid per minuut"
        },
        "Oxygen Saturation": {
            "value": patient.spo2,
            "type": "number",
            "description": "Zuurstof gehalte in bloed in procenten"
        },
        "Systolic Blood Pressure": {
            "value": patient.sbp,
            "type": "number",
            "description": "Systolic blood pressure van de patient in mmHg. (Bovenste bloeddruk)"
        },
        "Diastolic Blood Pressure": {
            "value": patient.dbp,
            "type": "number",
            "description": "Diastolic blood pressure van de patient in mmHg. (Onderste bloeddruk)"
        },
        "Mean Blood Pressure": {
            "value": patient.mbp,
            "type": "number",
            "description": "Mean blood pressure van de patient in mmHg. (Gemiddelde bloeddruk van een periode van tijd)"
        },
        "White Blood Cell Count": {
            "value": patient.wbc.toFixed(2),
            "type": "number",
            "description": "Hoeveelheid witte bloedcellen in 10^3/uL"
        },
        "Hemoglobin": {
            "value": patient.hemoglobin.toFixed(2),
            "type": "number",
            "description": "Hemoglobin gehalte in het bloed van de patient in g/dL. (Specifieke eiwit in het bloed dat zuurstof transporteert)"
        },
        "Platelet Count": {
            "value": patient.platelet,
            "type": "number",
            "description": "Aantal bloedplaatjes in het bloed van de patient in 10^3/uL"
        },
        "Blood Urea Nitrogen": {
            "value": patient.bun,
            "type": "number",
            "description": "Blood area nitrogen gehalte in het bloed van de patient in mg/dL. (Indicator van de nierfunctie)"
        },
        "Creatinine": {
            "value": patient.cr.toFixed(2),
            "type": "number",
            "description": "Creatine gehalte in het bloed van de patient in mg/dL."
        },
        "Glucose": {
            "value": patient.glu,
            "type": "number",
            "description": "Suiker gehalte in het bloed van de patient in mg/dL."
        },
        "Sodium": {
            "value": patient.Na,
            "type": "number",
            "description": "Natrium gehalte in het bloed van de patient in mmol/L."
        },
        "Chloride": {
            "value": patient.Cl,
            "type": "number",
            "description": "Chloride gehalte in het bloed van de patient in mmol/L."
        },
        "Potassium": {
            "value": patient.K.toFixed(2),
            "type": "number",
            "description": "Kalium gehalte in het bloed van de patient in mmol/L."
        },
        "Magnesium": {
            "value": patient.Mg,
            "type": "number",
            "description": "Magnesium gehalte in het bloed van de patient in mmol/L."
        },
        "Calcium": {
            "value": patient.Ca,
            "type": "number",
            "description": "Calcium gehalte in het bloed van de patient in mmol/L."
        },
        "Phosphorus": {
            "value": patient.P.toFixed(2),
            "type": "number",
            "description": "Fosfor gehalte in het bloed van de patient in mg/dL."
        },
        "International Normalized Ratio": {
            "value": patient.inr.toFixed(2),
            "type": "number",
            "description": "International normalized ratio van de patient. (bloedstolling test) (Hoelang het duurt voordat het bloed stolt. lager is beter)"
        },
        "Prothrombin Time": {
            "value": patient.pt.toFixed(2),
            "type": "number",
            "description": "Prothrombin time van de patient in seconden.(bloedstolling test)"
        },
        "Partial Thromboplastin Time": {
            "value": patient.ptt.toFixed(2),
            "type": "number",
            "description": "Partial thromboplastin time van de patient in seconden.(bloedstolling test)"
        },
        "Bicarbonate": {
            "value": patient.bicarbonate,
            "type": "number",
            "description": "Bicarbonaat gehalte in het bloed van de patient in mmol/L."
        },
        "Anion Gap": {
            "value": patient.aniongap,
            "type": "number",
            "description": "Anion gap van de patient in mmol/L. (Het verschil tussen de gemeten positieve en negatieve ionen in het bloed) (huh interessant)"
        },
        "Glasgow Coma Scale": {
            "value": patient.gcs,
            "type": "number",
            "description": "Glasgow Coma Scale score van de patient.(schaal van 3-15. hoe lager hoe slechter)"
        },
        "Ventilation": {
            "value": convertToBoolean(patient.vent),
            "type": "number",
            "description": "Of de patient aan de beademing ligt. (Ja/Nee)"
        },
        "Continuous Renal Replacement Therapy": {
            "value": convertToBoolean(patient.crrt),
            "type": "number",
            "description": "Of de patient aan de dialyse ligt.(Ja/Nee)"
        },
        "Vasoconstrictors": {
            "value": convertToBoolean(patient?.vaso),
            "type": "number",
            "description": "Of de patient aan de vasoconstrictors ligt. (Ja/Nee) (Verhoogt de bloeddruk)"
        },
        "Sedatives": {
            "value": convertToBoolean(patient?.seda),
            "type": "number",
            "description": "Of de patient aan de sedatives ligt. (Ja/Nee)"
        },
        "Sequential Organ Failure Assessment Score": {
            "value": patient?.sofa_score,
            "type": "number",
            "description": "Sequential Organ Failure Assessment score van de patient. van eerste opname (schaal van 0-24. hoe hoger hoe slechter)"
        },
        "Acute Myocardial Infarction": {
            "value": convertToBoolean(patient?.ami),
            "type": "number",
            "description": "Of de patient een Acute Myocardial Infarction heeft gehad.(Ja/Nee)"
        },
        "Chronic Kidney Disease": {
            "value": convertToBoolean(patient?.ckd),
            "type": "number",
            "description": "Of de patient een Chronic Kidney Disease heeft.(Ja/Nee)"
        },
        "Chronic Obstructive Pulmonary Disease": {
            "value": convertToBoolean(patient?.copd),
            "type": "number",
            "description": "Of de patient een Chronic Obstructive Pulmonary Disease heeft.(Ja/Nee)"
        },
        "Hypertension": {
            "value": convertToBoolean(patient?.hyperte),
            "type": "number",
            "description": "Of de patient een Hypertension heeft.(Ja/Nee)"
        },
        "Diabetes Mellitus": {
            "value": convertToBoolean(patient?.dm),
            "type": "number",
            "description": "Of de patient een Diabetes Mellitus heeft.(Ja/Nee)"
        },
        "Acute Kidney Injury": {
            "value": convertToBoolean(patient?.aki),
            "type": "number",
            "description": "Of de patient een Acute Kidney Injury heeft gehad.(Ja/Nee)"
        },
        "Stroke": {
            "value": convertToBoolean(patient?.stroke),
            "type": "number",
            "description": "Of de patient een Stroke heeft gehad.(Ja/Nee)"
        },
        "Aisan": {
            "value": convertToBoolean(patient?.AISAN),
            "type": "boolean",
            "description": "Etniciteit van de patient."
        },
        "Black": {
            "value": convertToBoolean(patient.BLACK),
            "type": "boolean",
            "description": "Etniciteit van de patient."
        },
        "Hispanic": {
            "value": convertToBoolean(patient?.HISPANIC),
            "type": "boolean",
            "description": "Etniciteit van de patient"
        },
        "Other": {
            "value": convertToBoolean(patient?.OTHER),
            "type": "boolean",
            "description": "Etniciteit van de patient."
        },
        "White": {
            "value": convertToBoolean(patient?.WHITE),
            "type": "boolean",
            "description": "Etniciteit van de patient."
        },
        "Unknown": {
            "value": convertToBoolean(patient?.unknown),
            "type": "boolean",
            "description": "Etniciteit van de patient."
        },
        "Coronary Care Unit": {
            "value": convertToBoolean(patient?.CCU),
            "type": "boolean",
            "description": "Of de patient op hardbewaking heeft gehad.(Ja/Nee)"
        },
        "Cardiovascular Intensive Care Unit": {
            "value": convertToBoolean(patient?.CVICU),
            "type": "boolean",
            "description": "Of de patient een verzorging van de Cardiovascular Intensive Care Unit heeft gehad.(Ja/Nee)"
        },
        "Mobile Intensive Care Unit": {
            "value": convertToBoolean(patient?.MICU),
            "type": "boolean",
            "description": "Of de patient vervoerd is door de mobiele intensive care unit.(Ja/Nee)"
        },
        "Mobile/Surgical Intensive Care Unit": {
            "value": convertToBoolean(patient?.["MICU/SICU"]),
            "type": "boolean",
            "description": "Of de patient een behandeling van de surgical intensive care unit heeft gehad tijdens het vervoer van de mobiele intensive care unit.(Ja/Nee)"
        },
        "Neonatal Intensive Care Unit": {
            "value": convertToBoolean(patient?.NICU),
            "type": "boolean",
            "description": "Of de patient een behandeling van de neonatal intensive care unit heeft gehad (is voor newborn babies).(Ja/Nee)"
        },
        "Surgical Intensive Care Unit": {
            "value": convertToBoolean(patient?.SICU),
            "type": "boolean",
            "description": "Of de patient een behandeling van de surgical intensive care unit heeft gehad.(Ja/Nee)"
        },
        "Trauma Intensive Care Unit": {
            "value": convertToBoolean(patient?.TSICU),
            "type": "boolean",
            "description": "Of de patient een verzorging van de trauma intensive care unit heeft gehad.(Ja/Nee)"
        }
    };


    return (
        <>
            <div className="content">
                <DashboardInfoCard cardInformation={cardInfo}/>
                <DashboardInput features={humanBodyFeatures}
                                onPredict={setStates} prediction={prediction}
                                prediction_present={predictionPresent}/>
            </div>
        </>
    );
}

export default Dashboard;
