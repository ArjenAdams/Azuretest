import {
    Card,
    CardBody,
    CardHeader,
    CardSubtitle,
    CardTitle,
    Col,
    Row,
    Table
} from "reactstrap";
import React from "react";

function convertToBoolean(boolean) {
    return boolean === 1;
}

function CounterfactualValues(data) {
    if (data === null || data === undefined) {
        return (
            <>
                <Card>
                    <CardHeader>
                        <CardTitle className="text-dark text-2rem">Veranderingen
                            voor andere voorspelling</CardTitle>
                        <CardSubtitle className="text-muted text-1rem">
                            De minimale veranderingen in de patiëntkenmerken
                            die leiden tot een andere voorspelling
                        </CardSubtitle>
                    </CardHeader>
                </Card>
            </>)
    }
    const counterfactuals = data.counterfactuals
    console.log("Counterfactuals")
    console.log(counterfactuals)
    console.log("patient")
    console.log(data)
    const prediction = data.prediction
    console.log("prediction")
    console.log(prediction)
    const patient = data?.patient;
    let gender = "F"
    // Gender converten naar getallen
    if (patient.gender === 1) {
        gender = "M"
    }
    const predPatient = prediction.prediction
    let predictionPatient;
    let certaintyPatient = prediction.confidence;
    if (certaintyPatient === null || certaintyPatient === undefined) {
        console.log("Certainty Patient is null or undefined!")
        return (
            <>
                <Card>
                    <CardHeader>
                        <CardTitle className="text-dark text-2rem">Veranderingen
                            voor andere voorspelling</CardTitle>
                        <CardSubtitle className="text-muted text-1rem">
                            De minimale veranderingen in de patiëntkenmerken
                            die leiden tot een andere voorspelling
                        </CardSubtitle>
                    </CardHeader>
                    <CardBody>
                        Loading
                    </CardBody>
                </Card>
            </>)
    }
    if (Number(predPatient) >= 0.5) {
        predictionPatient = "Ja"
    } else {
        predictionPatient = "Nee"
    }

    const ownPatient = {
        "Heeft SAD?": {
            "value": predictionPatient,
            "type": "text"
        },
        "Zekerheidsscore": {
            "value": certaintyPatient.toFixed(2),
            "type": "number"
        },
        "Gender": {
            "value": gender,
            "type": "text"
        },
        "Age": {
            "value": patient.age,
            "type": "number"
        },
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
    let counterfactuals_list = []
    const titles = Object.keys(ownPatient)
    let origineleWaarden = [];

    for (const title in ownPatient) {
        const value = ownPatient[title].value;
        const type = ownPatient[title].type;
        origineleWaarden.push([value, type])
    }
    let size = counterfactuals.length - 1
    let counter = []

    while (size >= 0) {
        let counterfactual = counterfactuals[size]

        let gender = "F"
        // Gender converten naar getallen
        if (counterfactual.gender === 1) {
            gender = "M"
        }

        const pred = counterfactual.pred
        let prediction;
        let certainty = counterfactual.reg;
        if (Number(pred) >= 0.5) {
            prediction = "Ja"
            certainty = certainty * 100
        } else {
            prediction = "Nee"
            certainty = 100 - certainty * 100
        }
        let counterfactualPatient = {
            "Heeft SAD?": {
                "value": prediction,
                "type": "text"
            },
            "Zekerheidsscore": {
                "value": certainty.toFixed(2),
                "type": "number"
            },
            "Gender": {
                "value": gender,
                "type": "text"
            },
            "Age": {
                "value": counterfactual.age,
                "type": "number"
            },
            "Weight": {
                "value": counterfactual.weight,
                "type": "number"
            },
            "Temperature": {
                "value": counterfactual.temperature,
                "type": "number"
            },
            "Heart Rate": {
                "value": counterfactual.heart_rate,
                "type": "number"
            },
            "Respiration Rate": {
                "value": counterfactual.resp_rate,
                "type": "number"
            },
            "Oxygen Saturation": {
                "value": counterfactual.spo2,
                "type": "number"
            },
            "Systolic Blood Pressure": {
                "value": counterfactual.sbp,
                "type": "number"
            },
            "Diastolic Blood Pressure": {
                "value": counterfactual.dbp,
                "type": "number"
            },
            "Mean Blood Pressure": {
                "value": counterfactual.mbp,
                "type": "number"
            },
            "White Blood Cell Count": {
                "value": counterfactual.wbc,
                "type": "number"
            },
            "Hemoglobin": {
                "value": counterfactual.hemoglobin,
                "type": "number"
            },
            "Platelet Count": {
                "value": counterfactual.platelet,
                "type": "number"
            },
            "Blood Urea Nitrogen": {
                "value": counterfactual.bun,
                "type": "number"
            },
            "Creatinine": {
                "value": counterfactual.cr,
                "type": "number"
            },
            "Glucose": {
                "value": counterfactual.glu,
                "type": "number"
            },
            "Sodium": {
                "value": counterfactual.Na,
                "type": "number"
            },
            "Chloride": {
                "value": counterfactual.Cl,
                "type": "number"
            },
            "Potassium": {
                "value": counterfactual.K,
                "type": "number"
            },
            "Magnesium": {
                "value": counterfactual.Mg,
                "type": "number"
            },
            "Calcium": {
                "value": counterfactual.Ca,
                "type": "number"
            },
            "Phosphorus": {
                "value": counterfactual.P,
                "type": "number"
            },
            "International Normalized Ratio": {
                "value": counterfactual.inr,
                "type": "number"
            },
            "Prothrombin Time": {
                "value": counterfactual.pt,
                "type": "number"
            },
            "Partial Thromboplastin Time": {
                "value": counterfactual.ptt,
                "type": "number"
            },
            "Bicarbonate": {
                "value": counterfactual.bicarbonate,
                "type": "number"
            },
            "Anion Gap": {
                "value": counterfactual.aniongap,
                "type": "number"
            },
            "Glasgow Coma Scale": {
                "value": counterfactual.gcs,
                "type": "number"
            },
            "Ventilation": {
                "value": counterfactual.vent,
                "type": "number"
            },
            "Continuous Renal Replacement Therapy": {
                "value": counterfactual.crrt,
                "type": "number"
            },
            "Vasoconstrictors": {
                "value": counterfactual?.vaso,
                "type": "number"
            },
            "Sedatives": {
                "value": counterfactual?.seda,
                "type": "number"
            },
            "Sequential Organ Failure Assessment Score": {
                "value": counterfactual?.sofa_score,
                "type": "number"
            },
            "Acute Myocardial Infarction": {
                "value": counterfactual?.ami,
                "type": "number"
            },
            "Chronic Kidney Disease": {
                "value": counterfactual?.ckd,
                "type": "number"
            },
            "Chronic Obstructive Pulmonary Disease": {
                "value": counterfactual?.copd,
                "type": "number"
            },
            "Hypertension": {
                "value": counterfactual?.hyperte,
                "type": "number"
            },
            "Diabetes Mellitus": {
                "value": counterfactual?.dm,
                "type": "number"
            },
            "Acute Kidney Injury": {
                "value": counterfactual?.aki,
                "type": "number"
            },
            "Stroke": {
                "value": counterfactual?.stroke,
                "type": "number"
            },
            "Aisan": {
                "value": convertToBoolean(counterfactual?.AISAN),
                "type": "boolean"
            },
            "Black": {
                "value": convertToBoolean(counterfactual.BLACK),
                "type": "boolean"
            },
            "Hispanic": {
                "value": convertToBoolean(counterfactual?.HISPANIC),
                "type": "boolean"
            },
            "Other": {
                "value": convertToBoolean(counterfactual?.OTHER),
                "type": "boolean"
            },
            "White": {
                "value": convertToBoolean(counterfactual?.WHITE),
                "type": "boolean"
            },
            "Unknown": {
                "value": convertToBoolean(counterfactual?.unknown),
                "type": "boolean"
            },
            "Coronary Care Unit": {
                "value": convertToBoolean(counterfactual?.CCU),
                "type": "boolean"
            },
            "Cardiovascular Intensive Care Unit": {
                "value": convertToBoolean(counterfactual?.CVICU),
                "type": "boolean"
            },
            "Mobile Intensive Care Unit": {
                "value": convertToBoolean(counterfactual?.MICU),
                "type": "boolean"
            },
            "Mobile/Surgical Intensive Care Unit": {
                "value": convertToBoolean(counterfactual?.MICU_SICU),
                "type": "boolean"
            },
            "Neonatal Intensive Care Unit": {
                "value": convertToBoolean(counterfactual?.NICU),
                "type": "boolean"
            },
            "Surgical Intensive Care Unit": {
                "value": convertToBoolean(counterfactual?.SICU),
                "type": "boolean"
            },
            "Trauma Intensive Care Unit": {
                "value": convertToBoolean(counterfactual?.TSICU),
                "type": "boolean"
            },
        }
        counterfactuals_list.push(counterfactualPatient)

        let counterfactualWaarden = [];
        for (const title in counterfactualPatient) {
            const value = counterfactualPatient[title].value;
            const type = counterfactualPatient[title].type;
            counterfactualWaarden.push([value, type])
        }
        counter.push(counterfactualWaarden)
        counterfactualWaarden = []
        size--
    }

    let predictionResult;
    let certaintyScore

    if (counterfactuals.length > 0) {
        const pred = counterfactuals[0]["pred"]
        certaintyScore = counterfactuals[0]["reg"]
        if (Number(pred) >= 0.5) {
            predictionResult = "Ja"
            certaintyScore = certaintyScore * 100
        } else {
            predictionResult = "Nee"
            certaintyScore = 100 - certaintyScore * 100
        }
    }

    const dummyData = titles.map((title, index) => ({
        id: index + 1,
        gegevens: title,
        origineleWaarde: origineleWaarden[index][0],
        typeInput: origineleWaarden[index][1],
        counterfactuals: counter ? counter.map(cf => cf[index][0]) : []
    }));


    function colorRow(row) {
        if (row.counterfactuals === row.origineleWaarde) {
            return "td-values";
        } else {
            return "row-color";
        }
    }

    return (
        <>
            <Card>
                <CardHeader>
                    <CardTitle className="text-dark text-2rem">Veranderingen
                        voor andere voorspelling</CardTitle>
                    <CardSubtitle className="text-muted text-1rem">
                        De minimale veranderingen in de patiëntkenmerken die
                        leiden tot een andere voorspelling
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
                                    {counter && counter.map((_, index) => (
                                        <th key={index}
                                            className="th-align-middle">Counterfactual {index + 1}</th>
                                    ))}

                                </tr>
                                </thead>
                                <tbody>
                                {dummyData.map((row) => (

                                    <tr key={row.id}>
                                        <td className={`${colorRow(row)} td-left-align`}>{row.gegevens}</td>
                                        {typeof (row.origineleWaarde) === "number" && row.origineleWaarde.toString().includes(".") ? (
                                            <td className={`${colorRow(row)} td-values`}>{Number.parseFloat(row.origineleWaarde).toFixed(2)}</td>
                                        ) : (
                                            <td className={`${colorRow(row)} td-values`}>{row.origineleWaarde.toString()}</td>
                                        )}
                                        {row.counterfactuals && row.counterfactuals.length > 0 ? (
                                            row.counterfactuals.map((value, index) => (
                                                typeof (value) === "number" && value.toString().includes(".") ? (
                                                    <td key={index} className={`${colorRow(row)} td-values`}>{Number.parseFloat(value).toFixed(2)}</td>
                                                ) : (
                                                    <td key={index} className={`${colorRow(row)} td-values`}>{value.toString()}</td>
                                                )
                                            ))
                                        ) : (
                                            <td className={`${colorRow(row)} td-values`}
                                                colSpan={counter.length}>No
                                                Data</td>
                                        )}
                                    </tr>

                                ))}
                                </tbody>
                            </Table>
                        </Col>
                    </Row>
                </CardBody>
            </Card>
        </>
    );
}

export default CounterfactualValues;