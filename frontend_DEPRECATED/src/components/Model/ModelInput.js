import {
    Card,
    CardBody,
    CardHeader,
    CardSubtitle,
    CardTitle,
    Col,
    Input,
    Row,
    Table
} from 'reactstrap';
import React, {useEffect, useState} from "react";
import API_URL from "../../handler/api";
import {fetchPredictionData} from "../../handler/sessionStorageHandler";
import ChangeValuesModal from "../Modals/ChangeValuesModal";
import PredictionChangedModal from "../Modals/PredictionChangedModal";

function convertToBoolean(boolean) {
    return boolean === 1;
}

function ModelInput({patient}) {
    const oldPrediction = fetchPredictionData();
    const [newPrediction, setNewPrediction] = useState({})
    const [predictionChangedIsOpen, setPredictionChangedIsOpen] = useState(false);
    const [valueChangedIsOpen, setValueChangedIsOpen] = useState(false);
    const [modalFeature, setModalFeature] = useState('')

    const humanBodyFeaturesNieuw = {
        "Weight": {
            "value": patient.weight,
            "type": "number"
        },
        "Temperature": {
            "value": patient.temperature,
            "type": "number"
        },
        "Heart Rate": {
            "value": patient.heart_rate,
            "type": "number"
        },
        "Respiration Rate": {
            "value": patient.resp_rate,
            "type": "number"
        },
        "Oxygen Saturation": {
            "value": patient.spo2,
            "type": "number"
        },
        "Systolic Blood Pressure": {
            "value": patient.sbp,
            "type": "number"
        },
        "Diastolic Blood Pressure": {
            "value": patient.dbp,
            "type": "number"
        },
        "Mean Blood Pressure": {
            "value": patient.mbp,
            "type": "number"
        },
        "White Blood Cell Count": {
            "value": patient.wbc,
            "type": "number"
        },
        "Hemoglobin": {
            "value": patient.hemoglobin,
            "type": "number"
        },
        "Platelet Count": {
            "value": patient.platelet,
            "type": "number"
        },
        "Blood Urea Nitrogen": {
            "value": patient.bun,
            "type": "number"
        },
        "Creatinine": {
            "value": patient.cr,
            "type": "number"
        },
        "Glucose": {
            "value": patient.glu,
            "type": "number"
        },
        "Sodium": {
            "value": patient.Na,
            "type": "number"
        },
        "Chloride": {
            "value": patient.Cl,
            "type": "number"
        },
        "Potassium": {
            "value": patient.K,
            "type": "number"
        },
        "Magnesium": {
            "value": patient.Mg,
            "type": "number"
        },
        "Calcium": {
            "value": patient.Ca,
            "type": "number"
        },
        "Phosphorus": {
            "value": patient.P,
            "type": "number"
        },
        "International Normalized Ratio": {
            "value": patient.inr,
            "type": "number"
        },
        "Prothrombin Time": {
            "value": patient.pt,
            "type": "number"
        },
        "Partial Thromboplastin Time": {
            "value": patient.ptt,
            "type": "number"
        },
        "Bicarbonate": {
            "value": patient.bicarbonate,
            "type": "number"
        },
        "Anion Gap": {
            "value": patient.aniongap,
            "type": "number"
        },
        "Glasgow Coma Scale": {
            "value": patient.gcs,
            "type": "number"
        },
        "Ventilation": {
            "value": patient.vent,
            "type": "number"
        },
        "Continuous Renal Replacement Therapy": {
            "value": patient.crrt,
            "type": "number"
        },
        "Vasoconstrictors": {
            "value": patient?.vaso,
            "type": "number"
        },
        "Sedatives": {
            "value": patient?.seda,
            "type": "number"
        },
        "Sequential Organ Failure Assessment Score": {
            "value": patient?.sofa_score,
            "type": "number"
        },
        "Acute Myocardial Infarction": {
            "value": patient?.ami,
            "type": "number"
        },
        "Chronic Kidney Disease": {
            "value": patient?.ckd,
            "type": "number"
        },
        "Chronic Obstructive Pulmonary Disease": {
            "value": patient?.copd,
            "type": "number"
        },
        "Hypertension": {
            "value": patient?.hyperte,
            "type": "number"
        },
        "Diabetes Mellitus": {
            "value": patient?.dm,
            "type": "number"
        },
        "Acute Kidney Injury": {
            "value": patient?.aki,
            "type": "number"
        },
        "Stroke": {
            "value": patient?.stroke,
            "type": "number"
        },
        "Aisan": {
            "value": convertToBoolean(patient?.AISAN),
            "type": "boolean"
        },
        "Black": {
            "value": convertToBoolean(patient.BLACK),
            "type": "boolean"
        },
        "Hispanic": {
            "value": convertToBoolean(patient?.HISPANIC),
            "type": "boolean"
        },
        "Other": {
            "value": convertToBoolean(patient?.OTHER),
            "type": "boolean"
        },
        "White": {
            "value": convertToBoolean(patient?.WHITE),
            "type": "boolean"
        },
        "Unknown": {
            "value": convertToBoolean(patient?.unknown),
            "type": "boolean"
        },
        "Coronary Care Unit": {
            "value": convertToBoolean(patient?.CCU),
            "type": "boolean"
        },
        "Cardiovascular Intensive Care Unit": {
            "value": convertToBoolean(patient?.CVICU),
            "type": "boolean"
        },
        "Mobile Intensive Care Unit": {
            "value": convertToBoolean(patient?.MICU),
            "type": "boolean"
        },
        "Mobile/Surgical Intensive Care Unit": {
            "value": convertToBoolean(patient?.MICU_SICU),
            "type": "boolean"
        },
        "Neonatal Intensive Care Unit": {
            "value": convertToBoolean(patient?.NICU),
            "type": "boolean"
        },
        "Surgical Intensive Care Unit": {
            "value": convertToBoolean(patient?.SICU),
            "type": "boolean"
        },
        "Trauma Intensive Care Unit": {
            "value": convertToBoolean(patient?.TSICU),
            "type": "boolean"
        }
    }

    const [updatedFeatures, setUpdatedFeatures] = useState(humanBodyFeaturesNieuw);
    useEffect(() => {
        if (!fetchPredictionData()) {
            setNewPrediction({'prediction': 0, 'confidence': 0});
        } else {
            const predictionData = fetchPredictionData();
            setNewPrediction({
                'prediction': predictionData['prediction'],
                'confidence': predictionData['confidence']
            });
        }
    }, []);


    const inputHandler = (event) => {
        const updatedFeature = event.target.title
        switch (typeof updatedFeatures[updatedFeature].value) {
            case "number":
                updatedFeatures[updatedFeature].value = Number(event.target.value)
                break;
            case "string":
                updatedFeatures[updatedFeature].value = String(event.target.value)
                break;
            //TODO: Dropdown voor Units en Etniciteit maken.
            case "boolean":
                if (event.target.value.toString().toLowerCase() === "true") {
                    updatedFeatures[updatedFeature].value = true
                    break;
                }
                if (event.target.value.toString().toLowerCase() === "false") {
                    updatedFeatures[updatedFeature].value = false
                    break;
                }
                break;
            default:
                updatedFeatures[updatedFeature].value = String(event.target.value)
                break;
        }
    }

    const translatePrediction = (prediction) => {
        if (Number(prediction) < 0.5) {
            return "Nee"
        } else if (Number(prediction) >= 0.5) {
            return "Ja"
        } else {
            return "Unknown"
        }
    }

    const predictionHandler = async () => {
        setValueChangedIsOpen(false);

        const response = await fetch(API_URL + "predict", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(
                {
                    "age": patient.age,
                    "weight": updatedFeatures['Weight'].value,
                    "gender": patient.gender,
                    "temperature": updatedFeatures['Temperature'].value,
                    "heart_rate": updatedFeatures['Heart Rate'].value,
                    "resp_rate": updatedFeatures['Respiration Rate'].value,
                    "spo2": updatedFeatures['Oxygen Saturation'].value,
                    "sbp": updatedFeatures['Systolic Blood Pressure'].value,
                    "dbp": updatedFeatures['Diastolic Blood Pressure'].value,
                    "mbp": updatedFeatures['Mean Blood Pressure'].value,
                    "wbc": updatedFeatures['White Blood Cell Count'].value,
                    "hemoglobin": updatedFeatures['Hemoglobin'].value,
                    "platelet": updatedFeatures['Platelet Count'].value,
                    "bun": updatedFeatures['Blood Urea Nitrogen'].value,
                    "cr": updatedFeatures['Creatinine'].value,
                    "glu": updatedFeatures['Glucose'].value,
                    "Na": updatedFeatures['Sodium'].value,
                    "Cl": updatedFeatures['Chloride'].value,
                    "K": updatedFeatures['Potassium'].value,
                    "Mg": updatedFeatures['Magnesium'].value,
                    "Ca": updatedFeatures['Calcium'].value,
                    "P": updatedFeatures['Phosphorus'].value,
                    "inr": updatedFeatures['International Normalized Ratio'].value,
                    "pt": updatedFeatures['Prothrombin Time'].value,
                    "ptt": updatedFeatures['Partial Thromboplastin Time'].value,
                    "bicarbonate": updatedFeatures['Bicarbonate'].value,
                    "aniongap": updatedFeatures['Anion Gap'].value,
                    "gcs": updatedFeatures['Glasgow Coma Scale'].value,
                    "vent": updatedFeatures['Ventilation'].value,
                    "crrt": updatedFeatures['Continuous Renal Replacement Therapy'].value,
                    "vaso": updatedFeatures['Vasoconstrictors'].value,
                    "seda": updatedFeatures['Sedatives'].value,
                    "sofa_score": updatedFeatures['Sequential Organ Failure Assessment Score'].value,
                    "ami": updatedFeatures['Acute Myocardial Infarction'].value,
                    "ckd": updatedFeatures['Chronic Kidney Disease'].value,
                    "copd": updatedFeatures['Chronic Obstructive Pulmonary Disease'].value,
                    "hyperte": updatedFeatures['Hypertension'].value,
                    "dm": updatedFeatures['Diabetes Mellitus'].value,
                    "aki": updatedFeatures['Acute Kidney Injury'].value,
                    "stroke": updatedFeatures['Stroke'].value,
                    "AISAN": updatedFeatures['Aisan'].value,
                    "BLACK": updatedFeatures['Black'].value,
                    "HISPANIC": updatedFeatures['Hispanic'].value,
                    "OTHER": updatedFeatures['Other'].value,
                    "WHITE": updatedFeatures['White'].value,
                    "unknown": updatedFeatures['Unknown'].value,
                    "CCU": updatedFeatures['Coronary Care Unit'].value,
                    "CVICU": updatedFeatures['Cardiovascular Intensive Care Unit'].value,
                    "MICU": updatedFeatures['Mobile Intensive Care Unit'].value,
                    "MICU/SICU": updatedFeatures['Mobile/Surgical Intensive Care Unit'].value,
                    "NICU": updatedFeatures['Neonatal Intensive Care Unit'].value,
                    "SICU": updatedFeatures['Surgical Intensive Care Unit'].value,
                    "TSICU": updatedFeatures['Trauma Intensive Care Unit'].value
                }
            )
        });
        const predictionString = await response.json();
        const prediction = JSON.parse(predictionString);

        setNewPrediction({
            'prediction': prediction.prediction,
            'confidence': prediction.confidence
        })

        if (oldPrediction.prediction !== prediction.prediction) {
            togglePredictionChangedModal();
        }
    }

    const titles = Object.keys(humanBodyFeaturesNieuw)

    let origineleWaarden = [];

    for (const title in humanBodyFeaturesNieuw) {
        const value = humanBodyFeaturesNieuw[title].value;
        const type = humanBodyFeaturesNieuw[title].type;
        origineleWaarden.push([value, type])
    }
    const data = titles.map((title, index) => ({
        id: index + 1,
        gegevens: title,
        origineleWaarde: origineleWaarden[index][0],
        typeInput: origineleWaarden[index][1],
        nieuweWaarde: updatedFeatures[title].value,
    }));

    function colorRow(row) {
        if (row.nieuweWaarde === row.origineleWaarde) {
            return "td-values";
        } else {
            return "row-color";
        }
    }

    const handleNextInputField = (e) => {
        getChangedFeatures();
        setModalFeature(e.target.title)
        if (e.key !== 'Enter') {
            return
        }
        let targetId = Number(e.target.id);
        const table = document.querySelector("#input-fields-table")
        const childrenNodes = table.children

        if (targetId === childrenNodes.length) {
            targetId = 0
        }

        // Focus on the next item in the input list
        childrenNodes.item(targetId).children.item(2).firstChild.focus()

        // Dont toggle the modal if the value hasn't changed
        const tempTitle = e.target.title

        if (humanBodyFeaturesNieuw[tempTitle].value === e.target.value) { // Using == instead of === for values that are 0
            return
        }
        toggleValueChangedModal();

    }

    const getChangedFeatures = () => {
        let changedFeatureTitles = [];
        titles.forEach((title) => {
            if (humanBodyFeaturesNieuw[title].value !== updatedFeatures[title].value) {
                changedFeatureTitles.push(title)
            }
        })

        return changedFeatureTitles
    }

    const toggleValueChangedModal = () => {
        setValueChangedIsOpen(!valueChangedIsOpen);
    }

    const togglePredictionChangedModal = () => {
        setPredictionChangedIsOpen(!predictionChangedIsOpen)
    }


    return (
        <>
            <ChangeValuesModal isOpen={valueChangedIsOpen}
                               toggle={toggleValueChangedModal}
                               makePrediction={predictionHandler}
                               feature={modalFeature}></ChangeValuesModal>
            <PredictionChangedModal isOpen={predictionChangedIsOpen}
                                    toggle={togglePredictionChangedModal}
                                    newPrediction={translatePrediction(newPrediction.prediction)}
                                    previousPrediction={translatePrediction(oldPrediction.prediction)}
                                    newCertaintyScore={newPrediction.confidence}
                                    previousCertaintyScore={oldPrediction.confidence}
                                    features={getChangedFeatures()}
            >
            </PredictionChangedModal>
            <Card>
                <CardHeader>
                    <CardTitle className="text-dark text-2rem">Patiëntkenmerken
                        wijzigen</CardTitle>
                    <CardSubtitle className="text-muted text-1rem">
                        Verander de patiëntkenmerken om te zien hoe de
                        voorspelling verandert
                    </CardSubtitle>
                </CardHeader>
                <CardBody>
                    <Row>
                        <Col md={8}>
                            <Table>
                                <thead>
                                <tr>
                                    <th>Gegevens</th>
                                    <th className="th-align-middle">Originele
                                        waarde
                                    </th>
                                    <th>Voer nieuwe waarde in</th>
                                </tr>
                                </thead>
                                <tbody id={"input-fields-table"}>
                                {data.map((row) => (
                                    <tr id={row.id} key={row.id}>
                                        <td className={`${colorRow(row)} td-left-align`}>

                                            {row.gegevens}
                                        </td>
                                        {typeof (row.origineleWaarde) === "number" && row.origineleWaarde.toString().includes(".") ? (
                                            <td className={`${colorRow(row)} td-values`}>{Number.parseFloat(row.origineleWaarde).toFixed(2)}</td>
                                        ) : (
                                            <td className={`${colorRow(row)} td-values`}>{row.origineleWaarde}</td>
                                        )}
                                        {typeof (row.nieuweWaarde) === "number" && row.nieuweWaarde.toString().includes(".") ? (
                                            <td className={`${colorRow(row)} td-values`}>
                                                <Input
                                                    id={row.id}
                                                    title={row.gegevens}
                                                    onKeyDown={handleNextInputField}
                                                    type={row.typeInput}
                                                    placeholder={"Enter " + row.typeInput}
                                                    defaultValue={Number.parseFloat(row.nieuweWaarde).toFixed(2)}
                                                    onChange={inputHandler}
                                                />
                                            </td>
                                        ) : (
                                            <td className={`${colorRow(row)} td-values`}>
                                                <Input
                                                    id={row.id}
                                                    title={row.gegevens}
                                                    onKeyDown={handleNextInputField}
                                                    type={row.typeInput}
                                                    placeholder={"Enter " + row.typeInput}
                                                    defaultValue={row.nieuweWaarde}
                                                    onChange={inputHandler}
                                                />
                                            </td>)}
                                    </tr>
                                ))}
                                </tbody>
                            </Table>
                        </Col>
                        <Col md={4}>
                            <CardSubtitle className="CardSubtitle_newPred">Nieuwe
                                voorspelling</CardSubtitle>
                            <Card>
                                <CardHeader>
                                    <div className="numbers">
                                        <p className="card-category text-blue font-weight-bolder">
                                            Heeft SAD?
                                        </p>
                                    </div>
                                </CardHeader>
                                <CardBody>
                                    <div className="numbers text-center">
                                        <CardTitle
                                            className="font-italic size1-5rem font-weight-bold">
                                            {translatePrediction(newPrediction.prediction)}
                                        </CardTitle>
                                    </div>
                                </CardBody>
                            </Card>
                            <Card>
                                <CardHeader>
                                    <div className="numbers">
                                        <CardTitle
                                            className="card-category font-weight-bold">Certainty
                                            Score</CardTitle>
                                    </div>
                                </CardHeader>
                                <CardBody>
                                    <div className="numbers text-center">
                                        <CardTitle
                                            className="font-italic size1-5rem font-weight-bold">
                                            {newPrediction.confidence}
                                        </CardTitle>
                                    </div>
                                </CardBody>
                            </Card>
                            <div className="text-center">
                                <button className="btn btn-primary"
                                        onClick={toggleValueChangedModal}>Nieuwe
                                    Voorspelling
                                    Opslaan
                                </button>
                            </div>
                        </Col>
                    </Row>
                </CardBody>
            </Card>
        </>
    );
}

export default ModelInput;
