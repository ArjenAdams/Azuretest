import {fetchPatientData} from "../handler/sessionStorageHandler";
import React, {useEffect, useState} from "react";
import Shap_Plots from "./Shap_Plots";
import API_URL from "../handler/api";


function Shap_Container() {
    const [patient, setPatient] = useState({});
    const [method, setMethod] = useState('BAR_PLOT');
    const [feature, setFeature] = useState('age');
    const [interactionIndex, setInteractionIndex] = useState('temperature');
    const [lijst_met_patienten, setLijstMetPatienten] = useState();
    const [shapResponse, setShapResponse] = useState("");
    const [range1_in_dataset, setRange1_in_dataset] = useState(0);
    const [range2_in_dataset, setRange2_in_dataset] = useState(50);
    const [hoeveelheid_features, setHoeveelheidFeatures] = useState(5);
    const handleChangeMethod = (event) => {
        setMethod(event.target.value);
        return (
            <>
                <div className="content">
                    <form onSubmit={handleSubmit}>
                        <label>Type Plot:</label>
                        <select value={method} onChange={handleChangeMethod}>
                            <option value="INTERACTIVE_FORCE_PLOT">INTERACTIVE
                                FORCE PLOT
                            </option>
                            <option value="BAR_PLOT">BAR PLOT</option>
                            <option value="BEESWARM_PLOT">BEESWARM PLOT
                            </option>
                            <option value="PPT_PLOT">PPT PLOT</option>
                            <option value="DEPENDENCE_PLOT">DEPENDENCE PLOT
                            </option>
                            <option value="LOCAL_FORCE_PLOT">LOCAL FORCE PLOT
                            </option>
                            <option value="LOCAL_WATERFALL_PLOT">LOCAL
                                WATERFALL PLOT
                            </option>
                            <option value="LOCAL_DECISION_PLOT">LOCAL DECISION
                                PLOT
                            </option>
                            <option value="LOCAL_MULTI_DECISION_PLOT">LOCAL
                                MULTI DECISION PLOT
                            </option>
                        </select>
                        {method === "INTERACTIVE_FORCE_PLOT" ? (
                            <div>
                                <label>Range1 in de dataset:</label>
                                <input type="number"
                                       value={range1_in_dataset}
                                       placeholder="Voer hier de eerste index van de range in!"
                                       onChange={handleChangeRange1}
                                />
                                <label>Range2 in de dataset:</label>
                                <input type="number"
                                       value={range2_in_dataset}
                                       placeholder="Voer hier de tweede index van de range in!"
                                       onChange={handleChangeRange2}
                                />
                            </div>
                        ) : (
                            <div></div>
                        )}
                        {method === "PPT_PLOT" || method === "DEPENDENCE_PLOT" ? (
                            <div>
                                <label>Type feature:</label>
                                <select value={feature}
                                        onChange={handleChangeFeature}>
                                    <option value="age">Age</option>
                                    <option value="weight">Weight</option>
                                    <option value="gender">Gender</option>
                                    <option value="temperature">Temperature
                                    </option>
                                    <option value="heart_rate">Heart Rate
                                    </option>
                                    <option value="resp_rate">Respiration
                                        Rate
                                    </option>
                                    <option value="spo2">SpO2</option>
                                    <option value="sbp">Systolic Blood
                                        Pressure
                                    </option>
                                    <option value="dbp">Diastolic Blood
                                        Pressure
                                    </option>
                                    <option value="mbp">Mean Blood Pressure
                                    </option>
                                    <option value="wbc">White Blood Cell
                                        Count
                                    </option>
                                    <option value="hemoglobin">Hemoglobin
                                    </option>
                                    <option value="platelet">Platelet Count
                                    </option>
                                    <option value="bun">Blood Urea Nitrogen
                                    </option>
                                    <option value="cr">Creatinine</option>
                                    <option value="glu">Glucose</option>
                                    <option value="Na">Sodium</option>
                                    <option value="Cl">Chloride</option>
                                    <option value="K">Potassium</option>
                                    <option value="Mg">Magnesium</option>
                                    <option value="Ca">Calcium</option>
                                    <option value="P">Phosphorus</option>
                                    <option value="inr">INR (International
                                        Normalized Ratio)
                                    </option>
                                    <option value="pt">Prothrombin Time
                                    </option>
                                    <option value="ptt">Partial Thromboplastin
                                        Time
                                    </option>
                                    <option value="bicarbonate">Bicarbonate
                                    </option>
                                    <option value="aniongap">Anion Gap</option>
                                    <option value="gcs">Glasgow Coma Scale
                                    </option>
                                    <option value="vent">Ventilator</option>
                                    <option value="crrt">Continuous Renal
                                        Replacement Therapy
                                    </option>
                                    <option value="vaso">Vasopressors</option>
                                    <option value="seda">Sedation</option>
                                    <option value="sofa_score">SOFA Score
                                    </option>
                                    <option value="ami">Acute Myocardial
                                        Infarction
                                    </option>
                                    <option value="ckd">Chronic Kidney
                                        Disease
                                    </option>
                                    <option value="copd">Chronic Obstructive
                                        Pulmonary Disease
                                    </option>
                                    <option value="hyperte">Hypertension
                                    </option>
                                    <option value="dm">Diabetes Mellitus
                                    </option>
                                    <option value="aki">Acute Kidney Injury
                                    </option>
                                    <option value="stroke">Stroke</option>
                                    <option value="AISAN">Asian</option>
                                    <option value="BLACK">Black</option>
                                    <option value="HISPANIC">Hispanic</option>
                                    <option value="OTHER">Other</option>
                                    <option value="WHITE">White</option>
                                    <option value="unknown">Unknown</option>
                                    <option value="CCU">Cardiac Care Unit
                                        (CCU)
                                    </option>
                                    <option value="CVICU">Cardiovascular
                                        Intensive Care Unit (CVICU)
                                    </option>
                                    <option value="MICU">Medical Intensive Care
                                        Unit (MICU)
                                    </option>
                                    <option value="MICU/SICU">Medical/Surgical
                                        Intensive Care Unit (MICU/SICU)
                                    </option>
                                    <option value="NICU">Neonatal Intensive
                                        Care Unit (NICU)
                                    </option>
                                    <option value="SICU">Surgical Intensive
                                        Care Unit (SICU)
                                    </option>
                                    <option value="TSICU">Trauma Surgical
                                        Intensive Care Unit (TSICU)
                                    </option>
                                </select>
                            </div>
                        ) : (
                            <div></div>
                        )}
                        {method === "DEPENDENCE_PLOT" ? (
                            <div>
                                <label>Type feature voor de interactie
                                    index:</label>
                                <select value={interactionIndex}
                                        onChange={handleChangeInteractionIndex}>
                                    <option value="age">Age</option>
                                    <option value="weight">Weight</option>
                                    <option value="gender">Gender</option>
                                    <option value="temperature">Temperature
                                    </option>
                                    <option value="heart_rate">Heart Rate
                                    </option>
                                    <option value="resp_rate">Respiration
                                        Rate
                                    </option>
                                    <option value="spo2">SpO2</option>
                                    <option value="sbp">Systolic Blood
                                        Pressure
                                    </option>
                                    <option value="dbp">Diastolic Blood
                                        Pressure
                                    </option>
                                    <option value="mbp">Mean Blood Pressure
                                    </option>
                                    <option value="wbc">White Blood Cell
                                        Count
                                    </option>
                                    <option value="hemoglobin">Hemoglobin
                                    </option>
                                    <option value="platelet">Platelet Count
                                    </option>
                                    <option value="bun">Blood Urea Nitrogen
                                    </option>
                                    <option value="cr">Creatinine</option>
                                    <option value="glu">Glucose</option>
                                    <option value="Na">Sodium</option>
                                    <option value="Cl">Chloride</option>
                                    <option value="K">Potassium</option>
                                    <option value="Mg">Magnesium</option>
                                    <option value="Ca">Calcium</option>
                                    <option value="P">Phosphorus</option>
                                    <option value="inr">INR (International
                                        Normalized Ratio)
                                    </option>
                                    <option value="pt">Prothrombin Time
                                    </option>
                                    <option value="ptt">Partial Thromboplastin
                                        Time
                                    </option>
                                    <option value="bicarbonate">Bicarbonate
                                    </option>
                                    <option value="aniongap">Anion Gap</option>
                                    <option value="gcs">Glasgow Coma Scale
                                    </option>
                                    <option value="vent">Ventilator</option>
                                    <option value="crrt">Continuous Renal
                                        Replacement Therapy
                                    </option>
                                    <option value="vaso">Vasopressors</option>
                                    <option value="seda">Sedation</option>
                                    <option value="sofa_score">SOFA Score
                                    </option>
                                    <option value="ami">Acute Myocardial
                                        Infarction
                                    </option>
                                    <option value="ckd">Chronic Kidney
                                        Disease
                                    </option>
                                    <option value="copd">Chronic Obstructive
                                        Pulmonary Disease
                                    </option>
                                    <option value="hyperte">Hypertension
                                    </option>
                                    <option value="dm">Diabetes Mellitus
                                    </option>
                                    <option value="aki">Acute Kidney Injury
                                    </option>
                                    <option value="stroke">Stroke</option>
                                    <option value="AISAN">Asian</option>
                                    <option value="BLACK">Black</option>
                                    <option value="HISPANIC">Hispanic</option>
                                    <option value="OTHER">Other</option>
                                    <option value="WHITE">White</option>
                                    <option value="unknown">Unknown</option>
                                    <option value="CCU">Cardiac Care Unit
                                        (CCU)
                                    </option>
                                    <option value="CVICU">Cardiovascular
                                        Intensive Care Unit (CVICU)
                                    </option>
                                    <option value="MICU">Medical Intensive Care
                                        Unit (MICU)
                                    </option>
                                    <option value="MICU/SICU">Medical/Surgical
                                        Intensive Care Unit (MICU/SICU)
                                    </option>
                                    <option value="NICU">Neonatal Intensive
                                        Care Unit (NICU)
                                    </option>
                                    <option value="SICU">Surgical Intensive
                                        Care Unit (SICU)
                                    </option>
                                    <option value="TSICU">Trauma Surgical
                                        Intensive Care Unit (TSICU)
                                    </option>
                                </select>
                            </div>
                        ) : (
                            <div></div>
                        )}
                        {method === "LOCAL_WATERFALL_PLOT" ? (
                            <div>
                                <label>Aantal features variabellen
                                    zien:</label>
                                <input type="number"
                                       value={hoeveelheid_features}
                                       placeholder="Voer hier het aantal features in!"
                                       onChange={handleChangeHoeveelheidFeatures}
                                />
                            </div>
                        ) : (
                            <div></div>
                        )}
                        {method === "LOCAL_MULTI_DECISION_PLOT" ? (
                            <div>
                                <label>Indexes van de te vergelijken patienten
                                    in array format: [index1, index2,
                                    ...]</label>
                                <input type={"text"}
                                       value={lijst_met_patienten}
                                       placeholder={"Voer hier een lijst in met patient ids/indexes!"}
                                       onChange={handleChangeLijstMetPatienten}
                                />
                            </div>
                        ) : (
                            <div></div>
                        )}
                        <button onClick={handleSubmit}>Submit</button>
                    </form>
                    {shapResponse !== undefined && shapResponse !== null ? (
                        <Shap_Plots shap_image={shapResponse}/>
                    ) : (
                        <p>Loading form data...</p>
                    )}
                </div>
            </>
        );
    };
    const handleChangeFeature = (event) => {
        setFeature(event.target.value);
    };
    const handleChangeLijstMetPatienten = (event) => {
        setLijstMetPatienten(event.target.value);
    };

    async function fetchData(form_data) {
        const response = await fetch(API_URL + '/generate_explanations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(form_data)
        });
        if (response.ok) {
            const shapData = await response.json();
            const shap = shapData.image
            setShapResponse(shap);

        } else {
            console.error("Error fetching SHAP data:", response.statusText);
        }
    }

    const handleChangeInteractionIndex = (event) => {
        setInteractionIndex(event.target.value);
    };

    const handleChangeRange1 = (event) => {
        const value = event.target.value;
        // Optional: Additional validation to ensure the value is a number
        if (!isNaN(value) && value !== '' && parseInt(value) > -1) {
            setRange1_in_dataset(parseInt(value));
        } else {
            window.alert("Geen geldige waarde ingevuld!\n\nDe waarde moet positief zijn!")
        }
    };
    const handleChangeRange2 = (event) => {
        const value = event.target.value;
        // Optional: Additional validation to ensure the value is a number
        if (!isNaN(value) && value !== '' && parseInt(value) > 0) {
            setRange2_in_dataset(parseInt(value));
        } else {
            window.alert("Geen geldige waarde ingevuld!\n\nDe waarde moet 0 of hoger zijn!")
        }
    };
    const handleChangeHoeveelheidFeatures = (event) => {
        const value = event.target.value;
        // Optional: Additional validation to ensure the value is a number
        if (!isNaN(value) && value !== '' && parseInt(value) > 0) {
            setHoeveelheidFeatures(parseInt(value));
        } else {
            window.alert("Geen geldige waarde ingevuld!\n\nDe waarde moet 0 of hoger zijn!")
        }
    };
    useEffect(() => {
        const fetchData = async () => {
            const patientData = await fetchPatientData();

            setPatient(patientData);
        };
        fetchData();
    }, []);

    const handleSubmit = (event) => {
        event.preventDefault();
        try {
            let id = 0
            if (patient !== undefined && patient !== null && patient.ID !== null) {
                id = patient.ID
            }
            let nieuw = []
            if (method === "LOCAL_MULTI_DECISION_PLOT") {
                if (lijst_met_patienten === undefined) {
                    throw new Error("lijst met patienten is niet gedefineerd!")
                }
                let oud = lijst_met_patienten.toString();
                if (oud.startsWith("[") && oud.endsWith("]")) {
                    oud = oud.replace("[", "")
                    oud = oud.replace("]", "")
                    let lijst = oud.split(",")
                    lijst.forEach((item) => {
                        item = parseInt(item)
                        if (!isNaN(item)) {
                            nieuw.push(item)
                        }
                    })
                } else {
                    throw new Error("de syntax van de lijst is incorrect!\n\nDe correcte syntax is:\n\t[index1,index2,index3]")
                }
                if (nieuw.length < 1) {
                    throw new Error("lijst met patienten is leeg!")
                }
            }
            if (method === "INTERACTIVE_FORCE_PLOT") {
                if (range1_in_dataset === undefined || range2_in_dataset === undefined) {
                    throw new Error("De beiden ranges moeten worden ingevuld!")
                }
                if (range1_in_dataset >= range2_in_dataset) {
                    throw new Error("Range1 mag niet groter of gelijk zijn aan Range2!")
                }
            }
            //TODO: Weghalen als de backend correct een afbeelding van de INTERACTIVE_FORCE_PLOT terug stuurd.
            if (method === "INTERACTIVE_FORCE_PLOT") {
                throw new Error("Het is nog niet mogelijk om een afbeelding van de shap grafiek:\n\tmet methode: "
                    + method + "\nvia de backend terug te sturen!\n\nDit wordt later nog geimplementeerd!")
            }

            let requestBody = {
                type_visualization: method,
                range: [range1_in_dataset, range2_in_dataset],
                feature_name: feature,
                interaction_index: interactionIndex,
                index: id,
                max_display: hoeveelheid_features,
                list_indices: nieuw
            };
            setShapResponse(fetchData(requestBody))
        } catch (Error) {
            window.alert(Error.message)
        }

    }
    return (
        <>
            <div className="content">
                <form onSubmit={handleSubmit}>
                    <label>Type Plot:</label>
                    <select value={method} onChange={handleChangeMethod}>
                        <option value="INTERACTIVE_FORCE_PLOT">INTERACTIVE
                            FORCE PLOT
                        </option>
                        <option value="BAR_PLOT">BAR PLOT</option>
                        <option value="BEESWARM_PLOT">BEESWARM PLOT</option>
                        <option value="PPT_PLOT">PPT PLOT</option>
                        <option value="DEPENDENCE_PLOT">DEPENDENCE PLOT
                        </option>
                        <option value="LOCAL_FORCE_PLOT">LOCAL FORCE PLOT
                        </option>
                        <option value="LOCAL_WATERFALL_PLOT">LOCAL WATERFALL
                            PLOT
                        </option>
                        <option value="LOCAL_DECISION_PLOT">LOCAL DECISION
                            PLOT
                        </option>
                        <option value="LOCAL_MULTI_DECISION_PLOT">LOCAL MULTI
                            DECISION PLOT
                        </option>
                    </select>
                    {method === "INTERACTIVE_FORCE_PLOT" ? (
                        <div>
                            <label>Range1 in de dataset:</label>
                            <input type="number"
                                   value={range1_in_dataset}
                                   placeholder="Voer hier de eerste index van de range in!"
                                   onChange={handleChangeRange1}
                            />
                            <label>Range2 in de dataset:</label>
                            <input type="number"
                                   value={range2_in_dataset}
                                   placeholder="Voer hier de tweede index van de range in!"
                                   onChange={handleChangeRange2}
                            />
                        </div>
                    ) : (
                        <div></div>
                    )}
                    {method === "PPT_PLOT" || method === "DEPENDENCE_PLOT" ? (
                        <div>
                            <label>Type feature:</label>
                            <select value={feature}
                                    onChange={handleChangeFeature}>
                                <option value="age">Age</option>
                                <option value="weight">Weight</option>
                                <option value="gender">Gender</option>
                                <option value="temperature">Temperature
                                </option>
                                <option value="heart_rate">Heart Rate</option>
                                <option value="resp_rate">Respiration Rate
                                </option>
                                <option value="spo2">SpO2</option>
                                <option value="sbp">Systolic Blood Pressure
                                </option>
                                <option value="dbp">Diastolic Blood Pressure
                                </option>
                                <option value="mbp">Mean Blood Pressure
                                </option>
                                <option value="wbc">White Blood Cell Count
                                </option>
                                <option value="hemoglobin">Hemoglobin</option>
                                <option value="platelet">Platelet Count
                                </option>
                                <option value="bun">Blood Urea Nitrogen
                                </option>
                                <option value="cr">Creatinine</option>
                                <option value="glu">Glucose</option>
                                <option value="Na">Sodium</option>
                                <option value="Cl">Chloride</option>
                                <option value="K">Potassium</option>
                                <option value="Mg">Magnesium</option>
                                <option value="Ca">Calcium</option>
                                <option value="P">Phosphorus</option>
                                <option value="inr">INR (International
                                    Normalized Ratio)
                                </option>
                                <option value="pt">Prothrombin Time</option>
                                <option value="ptt">Partial Thromboplastin
                                    Time
                                </option>
                                <option value="bicarbonate">Bicarbonate
                                </option>
                                <option value="aniongap">Anion Gap</option>
                                <option value="gcs">Glasgow Coma Scale</option>
                                <option value="vent">Ventilator</option>
                                <option value="crrt">Continuous Renal
                                    Replacement Therapy
                                </option>
                                <option value="vaso">Vasopressors</option>
                                <option value="seda">Sedation</option>
                                <option value="sofa_score">SOFA Score</option>
                                <option value="ami">Acute Myocardial
                                    Infarction
                                </option>
                                <option value="ckd">Chronic Kidney Disease
                                </option>
                                <option value="copd">Chronic Obstructive
                                    Pulmonary Disease
                                </option>
                                <option value="hyperte">Hypertension</option>
                                <option value="dm">Diabetes Mellitus</option>
                                <option value="aki">Acute Kidney Injury
                                </option>
                                <option value="stroke">Stroke</option>
                                <option value="AISAN">Asian</option>
                                <option value="BLACK">Black</option>
                                <option value="HISPANIC">Hispanic</option>
                                <option value="OTHER">Other</option>
                                <option value="WHITE">White</option>
                                <option value="unknown">Unknown</option>
                                <option value="CCU">Cardiac Care Unit (CCU)
                                </option>
                                <option value="CVICU">Cardiovascular Intensive
                                    Care Unit (CVICU)
                                </option>
                                <option value="MICU">Medical Intensive Care
                                    Unit (MICU)
                                </option>
                                <option value="MICU/SICU">Medical/Surgical
                                    Intensive Care Unit (MICU/SICU)
                                </option>
                                <option value="NICU">Neonatal Intensive Care
                                    Unit (NICU)
                                </option>
                                <option value="SICU">Surgical Intensive Care
                                    Unit (SICU)
                                </option>
                                <option value="TSICU">Trauma Surgical Intensive
                                    Care Unit (TSICU)
                                </option>
                            </select>
                        </div>
                    ) : (
                        <div></div>
                    )}
                    {method === "DEPENDENCE_PLOT" ? (
                        <div>
                            <label>Type feature voor de interactie
                                index:</label>
                            <select value={interactionIndex}
                                    onChange={handleChangeInteractionIndex}>
                                <option value="age">Age</option>
                                <option value="weight">Weight</option>
                                <option value="gender">Gender</option>
                                <option value="temperature">Temperature
                                </option>
                                <option value="heart_rate">Heart Rate</option>
                                <option value="resp_rate">Respiration Rate
                                </option>
                                <option value="spo2">SpO2</option>
                                <option value="sbp">Systolic Blood Pressure
                                </option>
                                <option value="dbp">Diastolic Blood Pressure
                                </option>
                                <option value="mbp">Mean Blood Pressure
                                </option>
                                <option value="wbc">White Blood Cell Count
                                </option>
                                <option value="hemoglobin">Hemoglobin</option>
                                <option value="platelet">Platelet Count
                                </option>
                                <option value="bun">Blood Urea Nitrogen
                                </option>
                                <option value="cr">Creatinine</option>
                                <option value="glu">Glucose</option>
                                <option value="Na">Sodium</option>
                                <option value="Cl">Chloride</option>
                                <option value="K">Potassium</option>
                                <option value="Mg">Magnesium</option>
                                <option value="Ca">Calcium</option>
                                <option value="P">Phosphorus</option>
                                <option value="inr">INR (International
                                    Normalized Ratio)
                                </option>
                                <option value="pt">Prothrombin Time</option>
                                <option value="ptt">Partial Thromboplastin
                                    Time
                                </option>
                                <option value="bicarbonate">Bicarbonate
                                </option>
                                <option value="aniongap">Anion Gap</option>
                                <option value="gcs">Glasgow Coma Scale</option>
                                <option value="vent">Ventilator</option>
                                <option value="crrt">Continuous Renal
                                    Replacement Therapy
                                </option>
                                <option value="vaso">Vasopressors</option>
                                <option value="seda">Sedation</option>
                                <option value="sofa_score">SOFA Score</option>
                                <option value="ami">Acute Myocardial
                                    Infarction
                                </option>
                                <option value="ckd">Chronic Kidney Disease
                                </option>
                                <option value="copd">Chronic Obstructive
                                    Pulmonary Disease
                                </option>
                                <option value="hyperte">Hypertension</option>
                                <option value="dm">Diabetes Mellitus</option>
                                <option value="aki">Acute Kidney Injury
                                </option>
                                <option value="stroke">Stroke</option>
                                <option value="AISAN">Asian</option>
                                <option value="BLACK">Black</option>
                                <option value="HISPANIC">Hispanic</option>
                                <option value="OTHER">Other</option>
                                <option value="WHITE">White</option>
                                <option value="unknown">Unknown</option>
                                <option value="CCU">Cardiac Care Unit (CCU)
                                </option>
                                <option value="CVICU">Cardiovascular Intensive
                                    Care Unit (CVICU)
                                </option>
                                <option value="MICU">Medical Intensive Care
                                    Unit (MICU)
                                </option>
                                <option value="MICU/SICU">Medical/Surgical
                                    Intensive Care Unit (MICU/SICU)
                                </option>
                                <option value="NICU">Neonatal Intensive Care
                                    Unit (NICU)
                                </option>
                                <option value="SICU">Surgical Intensive Care
                                    Unit (SICU)
                                </option>
                                <option value="TSICU">Trauma Surgical Intensive
                                    Care Unit (TSICU)
                                </option>
                            </select>
                        </div>
                    ) : (
                        <div></div>
                    )}
                    {method === "LOCAL_WATERFALL_PLOT" ? (
                        <div>
                            <label>Aantal features variabellen zien:</label>
                            <input type="number"
                                   value={hoeveelheid_features}
                                   placeholder="Voer hier het aantal features in!"
                                   onChange={handleChangeHoeveelheidFeatures}
                            />
                        </div>
                    ) : (
                        <div></div>
                    )}
                    {method === "LOCAL_MULTI_DECISION_PLOT" ? (
                        <div>
                            <label>Indexes van de te vergelijken patienten in
                                array format: [index1, index2, ...]</label>
                            <input type={"text"}
                                   value={lijst_met_patienten}
                                   placeholder={"Voer hier een lijst in met patient ids/indexes!"}
                                   onChange={handleChangeLijstMetPatienten}
                            />
                        </div>
                    ) : (
                        <div></div>
                    )}
                    <button onClick={handleSubmit}>Submit</button>
                </form>
                {shapResponse !== undefined && shapResponse !== null ? (
                    <Shap_Plots shap_image={shapResponse}/>
                ) : (
                    <p>Loading form data...</p>
                )}
            </div>
        </>
    );
}

export default Shap_Container;