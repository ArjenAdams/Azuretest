import {Col, Row, Button} from "reactstrap";
import Similar_Patients_Table
    from "../components/Similar Patiënts/Similar_Patiënts_Tabel";
import DashboardInfoCard
    from "../components/InformationCards/DashboardInfoCard";
import React, {useEffect, useState} from "react";
import {
    fetchPatientData,
    fetchPredictionData,
    fetchSimilarPatientsData,
    setSimilarPatientsData
} from "../handler/sessionStorageHandler";
import AdvancedFilter from "../components/Similar Patiënts/Filter/AdvancedFilter";

function Similar_Patients() {
    const [ownPatient, setOwnPatient] = useState({});
    const [prediction, setPrediction] = useState({});
    const [similarPatients, setSimilarPatients] = useState([]);
    const [hoeveelheid_similar_patients, setHoeveelheidSimilarPatients] = useState(2);

    const handleChangeHoeveelheid = (event) => {
        const value = event.target.value;
        // Optional: Additional validation to ensure the value is a number
        if (!isNaN(value) && value !== '') {
            setHoeveelheidSimilarPatients(parseInt(value));
        }
    };
    useEffect(() => {
        const fetchData = async () => {
            const patientData = await fetchPatientData();
            setOwnPatient(patientData);
            const predictionData = await fetchPredictionData();
            setPrediction(predictionData)
            await getSimilarPatients(patientData);
            const similarPatientsData = await fetchSimilarPatientsData();
            setSimilarPatients(similarPatientsData)
        };
        fetchData();
    }, []);

    const cardInfo = {
        "Patient ID": ownPatient?.ID || "",
        "Prediction": prediction.prediction || 0,
        "Probability": prediction.confidence || 0
    };
    const getSimilarPatients = async (patientData) => {
        if (patientData !== undefined) {
            console.log("Fetching similar patient")
            console.log(patientData)

            let gender = 0
            // Gender converten naar getallen
            if (patientData['gender'].value === "M") {
                gender = 1
            }
            const response = await fetch("http://localhost:5000/get_similarity", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    "ID": patientData["ID"],
                    "amount": hoeveelheid_similar_patients,
                    "filter": null
                }),
            });
            const similarPatientsData = await response.json();
            setSimilarPatientsData(similarPatientsData);
            setSimilarPatients(similarPatientsData);
            console.log("SimilarPatients:")
            console.log(similarPatientsData)
            return similarPatientsData;
        }
    }

    async function getSimilarPatientsFunction(filter) {
        let patientData = ownPatient

        if (patientData !== undefined) {
            console.log("Fetching similar patient")
            console.log(patientData)

            const response = await fetch("http://localhost:5000/get_similarity", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  "ID": patientData["ID"],
                    "amount": hoeveelheid_similar_patients,
                    "filter": filter
                }),
            });
            const similarPatientsData = await response.json();
            setSimilarPatientsData(similarPatientsData);
            setSimilarPatients(similarPatientsData);
            console.log("SimilarPatients:")
            console.log(similarPatientsData)
            return similarPatientsData;
        }
    }


  const handleSubmit = (event, filter = []) => {
        getSimilarPatientsFunction(filter);
    }

      const handleSubmitDefault = (event) => {

        getSimilarPatientsFunction(null);
    }


    const [modal, setModal] = useState(false);
    const toggleModal = () => setModal(!modal);
    return (
        <>
            <div className="content">
                <Row>
                    <Col md={12}>
                        <DashboardInfoCard cardInformation={cardInfo}/>
                        <form onSubmit={handleSubmitDefault}>
                            <label>Aantal Similar Patients:</label>
                            <input type="number"
                                   value={hoeveelheid_similar_patients}
                                   placeholder="Voer hier het aantal similar patients in!"
                                   onChange={handleChangeHoeveelheid}/>
                            <button onClick={handleSubmitDefault}>Submit</button>
                        </form>
                        <Button color="primary" type="button" style={{marginRight: '20px'}} onClick={toggleModal} className="mt-3">Filter</Button>
                            <AdvancedFilter modal={modal} toggleModal={toggleModal} handleSubmit={handleSubmit}></AdvancedFilter>
                        {similarPatients !== undefined && similarPatients.patients !== undefined && similarPatients.patients.length > 0 ? (
                            <Similar_Patients_Table
                                similar_patients={similarPatients}
                                own_patient={ownPatient}
                                prediction={prediction.confidence}/>
                        ) : (
                            <p>Loading patient data...</p>
                        )}
                    </Col>
                </Row>
            </div>
        </>
    );
}

export default Similar_Patients;