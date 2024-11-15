import CounterfactualValues
    from "../components/Counterfactual/CounterfactualValues";
import DashboardInfoCard
    from "../components/InformationCards/DashboardInfoCard";
import React, {useEffect, useState} from "react";
import {
    fetchCounterfactualData,
    fetchPatientData,
    fetchPredictionData,
    setCounterfactualData
} from "../handler/sessionStorageHandler";

function Counterfactual() {
    const [counterfactuals, setCounterfactuals] = useState({});
    const [patient, setPatient] = useState({});
    const [prediction, setPrediction] = useState({});
    const [method, setMethod] = useState('GENETIC');
    const handleChangeMethod = (event) => {
        setMethod(event.target.value);
    };
    const [hoeveelheid_counterfactuals, setHoeveelheid_counterfactuals] = useState(2);
    const handleChangeHoeveelheid = (event) => {
        const value = event.target.value;
        // Optional: Additional validation to ensure the value is a number
        if (!isNaN(value) && value !== '') {
            setHoeveelheid_counterfactuals(parseInt(value));
        }
    };

    useEffect(() => {
        const fetchData = async () => {
            const patientData = await fetchPatientData();
            setPatient(patientData);
            const predictionData = await fetchPredictionData();
            setPrediction(predictionData)
            await getCounterfactuals(patientData);
            const counterfactualData = await fetchCounterfactualData();
            setCounterfactuals(counterfactualData);
        };
        fetchData();
    }, []);

    const cardInfo = {
        "Patient ID": patient?.ID || "",
        "Prediction": prediction.prediction || 0,
        "Probability": prediction.confidence || 0
    };

    const getCounterfactuals = async (patientData) => {
        try {
            console.log("Fetching counterfactuals")
            console.log(patientData)
            console.log(hoeveelheid_counterfactuals)
            console.log(method)
            let gender = 0
            console.log(patient)
            // Gender converten naar getallen
            if (patientData['gender'].value === "M") {
                gender = 1
            }
            const response = await fetch("http://localhost:5000/counterfactual", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    "data": {
                        "age": patientData.age,
                        "weight": patientData.weight,
                        "gender": gender,
                        "temperature": patientData.temperature,
                        "heart_rate": patientData.heart_rate,
                        "resp_rate": patientData.resp_rate,
                        "spo2": patientData.spo2,
                        "sbp": patientData.sbp,
                        "dbp": patientData.dbp,
                        "mbp": patientData.mbp,
                        "wbc": patientData.wbc,
                        "hemoglobin": patientData.hemoglobin,
                        "platelet": patientData.platelet,
                        "bun": patientData.bun,
                        "cr": patientData.cr,
                        "glu": patientData.glu,
                        "Na": patientData.Na,
                        "Cl": patientData.Cl,
                        "K": patientData.K,
                        "Mg": patientData.Mg,
                        "Ca": patientData.Ca,
                        "P": patientData.P,
                        "inr": patientData.inr,
                        "pt": patientData.pt,
                        "ptt": patientData.ptt,
                        "bicarbonate": patientData.bicarbonate,
                        "aniongap": patientData.aniongap,
                        "gcs": patientData.gcs,
                        "vent": patientData.vent,
                        "crrt": patientData.crrt,
                        "vaso": patientData.vaso,
                        "seda": patientData.seda,
                        "sofa_score": patientData.sofa_score,
                        "ami": patientData.ami,
                        "ckd": patientData.ckd,
                        "copd": patientData.copd,
                        "hyperte": patientData.hyperte,
                        "dm": patientData.dm,
                        "aki": patientData.aki,
                        "stroke": patientData.stroke,
                        "AISAN": patientData.AISAN,
                        "BLACK": patientData.BLACK,
                        "HISPANIC": patientData.HISPANIC,
                        "OTHER": patientData.OTHER,
                        "WHITE": patientData.WHITE,
                        "unknown": patientData.unknown,
                        "CCU": patientData.CCU,
                        "CVICU": patientData.CVICU,
                        "MICU": patientData.MICU,
                        "MICU/SICU": patientData["MICU/SICU"],
                        "NICU": patientData.NICU,
                        "SICU": patientData.SICU,
                        "TSICU": patientData.TSICU
                    }, "method": method,
                    "amount": hoeveelheid_counterfactuals
                }),
            });
            const counterfactuals = await response.json();
            if (response.status !== 200) {
                throw new Error(counterfactuals)
            }
            setCounterfactualData(counterfactuals);
            setCounterfactuals(counterfactuals);
            console.log("Counterfactuals:")
            console.log(counterfactuals)
            return counterfactuals;
        } catch (Error) {
            console.log(Error)
            window.alert(Error)
        }

    }


    async function getCounter() {
        try {
            let patientData = patient
            console.log("Fetching counterfactuals")
            console.log(patientData)
            console.log(hoeveelheid_counterfactuals)
            console.log(method)
            let gender = 0
            console.log(patient)
            // Gender converten naar getallen
            if (patientData['gender'].value === "M") {
                gender = 1
            }
            const response = await fetch("http://localhost:5000/counterfactual", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    "data": {
                        "age": patientData.age,
                        "weight": patientData.weight,
                        "gender": gender,
                        "temperature": patientData.temperature,
                        "heart_rate": patientData.heart_rate,
                        "resp_rate": patientData.resp_rate,
                        "spo2": patientData.spo2,
                        "sbp": patientData.sbp,
                        "dbp": patientData.dbp,
                        "mbp": patientData.mbp,
                        "wbc": patientData.wbc,
                        "hemoglobin": patientData.hemoglobin,
                        "platelet": patientData.platelet,
                        "bun": patientData.bun,
                        "cr": patientData.cr,
                        "glu": patientData.glu,
                        "Na": patientData.Na,
                        "Cl": patientData.Cl,
                        "K": patientData.K,
                        "Mg": patientData.Mg,
                        "Ca": patientData.Ca,
                        "P": patientData.P,
                        "inr": patientData.inr,
                        "pt": patientData.pt,
                        "ptt": patientData.ptt,
                        "bicarbonate": patientData.bicarbonate,
                        "aniongap": patientData.aniongap,
                        "gcs": patientData.gcs,
                        "vent": patientData.vent,
                        "crrt": patientData.crrt,
                        "vaso": patientData.vaso,
                        "seda": patientData.seda,
                        "sofa_score": patientData.sofa_score,
                        "ami": patientData.ami,
                        "ckd": patientData.ckd,
                        "copd": patientData.copd,
                        "hyperte": patientData.hyperte,
                        "dm": patientData.dm,
                        "aki": patientData.aki,
                        "stroke": patientData.stroke,
                        "AISAN": patientData.AISAN,
                        "BLACK": patientData.BLACK,
                        "HISPANIC": patientData.HISPANIC,
                        "OTHER": patientData.OTHER,
                        "WHITE": patientData.WHITE,
                        "unknown": patientData.unknown,
                        "CCU": patientData.CCU,
                        "CVICU": patientData.CVICU,
                        "MICU": patientData.MICU,
                        "MICU/SICU": patientData["MICU/SICU"],
                        "NICU": patientData.NICU,
                        "SICU": patientData.SICU,
                        "TSICU": patientData.TSICU
                    }, "method": method,
                    "amount": hoeveelheid_counterfactuals
                }),
            });
            const counterfactuals = await response.json();

            if (response.status !== 200) {
                throw new Error(counterfactuals)
            }
            setCounterfactualData(counterfactuals);
            setCounterfactuals(counterfactuals);
            console.log("Counterfactuals:")
            console.log(counterfactuals)
            return counterfactuals;
        } catch (Error) {
            console.log(Error)
            window.alert(Error)
        }
    }

    const handleSubmit = (event) => {
        event.preventDefault();
        getCounter();
    }

    return (
        <>
            <div className="content">
                <DashboardInfoCard cardInformation={cardInfo}/>
                <form onSubmit={handleSubmit}>
                    <label>Type counterfactual:</label>
                    <select value={method} onChange={handleChangeMethod}>
                        <option value="RANDOM">RANDOM</option>
                        <option value="GENETIC">GENETIC</option>
                        <option value="KDTREE">KDTREE</option>
                    </select>
                    <label>Aantal counterfactuals:</label>
                    <input type="number"
                           value={hoeveelheid_counterfactuals}
                           placeholder="Voer hier het aantal counterfactuals in!"
                           onChange={handleChangeHoeveelheid}/>
                    <button onClick={handleSubmit}>Submit</button>
                </form>
                {patient.ID !== undefined && counterfactuals !== undefined ? (
                    <CounterfactualValues patient={patient}
                                          counterfactuals={counterfactuals}
                                          prediction={prediction}/>
                ) : (
                    <p>Loading patient data...</p>
                )}
            </div>
        </>
    );

}

export default Counterfactual;
