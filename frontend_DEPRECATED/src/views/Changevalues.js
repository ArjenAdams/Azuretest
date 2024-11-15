import ModelInput from "../components/Model/ModelInput";
import DashboardInfoCard
    from "../components/InformationCards/DashboardInfoCard";
import {
    fetchPatientData,
    fetchPredictionData
} from "../handler/sessionStorageHandler";
import {useEffect, useState} from "react";


function Changevalues() {
    const [patient, setPatient] = useState({});
    const [prediction, setPrediction] = useState({});

    useEffect(() => {
        const fetchData = async () => {
            const patientData = await fetchPatientData();

            setPatient(patientData);

            const predictionData = await fetchPredictionData();
            setPrediction(predictionData)
        };
        fetchData();
    }, []);
    console.log("prediction changevalues");
    console.log(prediction)
    const cardInfo = {
        "Patient ID": patient?.ID || "",
        "Prediction": prediction.prediction || 0,
        "Probability": prediction.confidence || 0
    };

    return (
        <>
            <div className="content">
                <DashboardInfoCard cardInformation={cardInfo}/>
                {patient.ID !== undefined ? (
                    <ModelInput patient={patient}/>
                ) : (
                    <p>Loading patient data...</p>
                )}
            </div>
        </>
    );
}

export default Changevalues;