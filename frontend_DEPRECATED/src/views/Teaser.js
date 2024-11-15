import {Button, Card, CardBody, CardHeader, Col, Row, Table} from "reactstrap";
import React, {useEffect, useState} from "react";
import {Link} from "react-router-dom";
import {
    fetchPatientData,
    setPredictionData
} from "../handler/sessionStorageHandler";
import API_URL from "../handler/api";
import DashboardInfoCard
    from "../components/InformationCards/DashboardInfoCard";

function Teaser() {
    const [patient, setPatient] = useState({});
    const [currentPrediction, setCurrentPrediction] = useState({});

    useEffect(() => {
        if (fetchPatientData() === null) {
            setPatient(null)
            return
        }
        const patientData = fetchPatientData();

        setPatient(patientData);
        // Setting the prediction globally using local storage so that every screen has acces to it
        const setPrediction = async () => {
            const response = await fetch(API_URL + "predict", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    'age': patientData.age,
                    'weight': patientData.weight,
                    'gender': patientData.gender,
                    'temperature': patientData.temperature,
                    'heart_rate': patientData.heart_rate,
                    'resp_rate': patientData.resp_rate,
                    'spo2': patientData.spo2,
                    'sbp': patientData.sbp,
                    'dbp': patientData.dbp,
                    'mbp': patientData.mbp,
                    'wbc': patientData.wbc,
                    'hemoglobin': patientData.hemoglobin,
                    'platelet': patientData.platelet,
                    'bun': patientData.bun,
                    'cr': patientData.cr,
                    'glu': patientData.glu,
                    'Na': patientData.Na,
                    'Cl': patientData.Cl,
                    'K': patientData.K,
                    'Mg': patientData.Mg,
                    'Ca': patientData.Ca,
                    'P': patientData.P,
                    'inr': patientData.inr,
                    'pt': patientData.pt,
                    'ptt': patientData.ptt,
                    'bicarbonate': patientData.bicarbonate,
                    'aniongap': patientData.aniongap,
                    'gcs': patientData.gcs,
                    'vent': patientData.vent,
                    'crrt': patientData.crrt,
                    'vaso': patientData.vaso,
                    'seda': patientData.seda,
                    'sofa_score': patientData.sofa_score,
                    'ami': patientData.ami,
                    'ckd': patientData.ckd,
                    'copd': patientData.copd,
                    'hyperte': patientData.hyperte,
                    'dm': patientData.dm,
                    'aki': patientData.aki,
                    'stroke': patientData.stroke,
                    'AISAN': patientData.AISAN,
                    'BLACK': patientData.BLACK,
                    'HISPANIC': patientData.HISPANIC,
                    'OTHER': patientData.OTHER,
                    'WHITE': patientData.WHITE,
                    'unknown': patientData.unknown,
                    'CCU': patientData.CCU,
                    'CVICU': patientData.CVICU,
                    'MICU': patientData.MICU,
                    'MICU/SICU': patientData['MICU/SICU'],
                    'NICU': patientData.NICU,
                    'SICU': patientData.SICU,
                    'TSICU': patientData.TSICU
                })
            });
            const predictionString = await response.json()
            const prediction = JSON.parse(predictionString);
            setPredictionData(prediction);
            setCurrentPrediction(prediction)
        }
        setPrediction();
    }, []);

    if (patient === null) {
        return (
            <div className={"content flex-column center"}>
                <h1 className={"text-dark font-weight-bold text-2rem"}>Laad
                    eerst een patiënt in</h1>
                <Link to={'/admin/patients'}>
                    <Button color={"primary"}>Patient Inladen</Button>
                </Link>
            </div>)
    }

    const cardInfo = {
        "Patient ID": patient.ID,
        "Prediction": currentPrediction.prediction,
        "Probability": currentPrediction.confidence
    };

    return (
        <div className={"content"}>
            <DashboardInfoCard cardInformation={cardInfo}></DashboardInfoCard>
            <Card className={"h-100"}>
                <CardHeader>
                    <div className="numbers text-center float-left">
                        <p className="card-category font-weight-bold">Globale
                            Informatie</p>
                    </div>
                </CardHeader>
                <CardBody>
                    <Card>
                        <CardHeader>
                            <div className="numbers text-center float-left">
                                <p className="card-title font-weight-bold">Algemene
                                    informatie over het predictiemodel</p>
                            </div>
                        </CardHeader>
                        <CardBody>
                            <Link to={'/admin/overview'}>
                                <Button color={"primary"}>
                                    <div>Global Screen</div>
                                    <div
                                        className={"nc-icon nc-chart-bar-32"}></div>
                                </Button>
                            </Link>
                        </CardBody>
                    </Card>
                </CardBody>
            </Card>
            <Card className={"h-100"}>
                <CardHeader>
                    <div className="numbers text-center float-left">
                        <p className="card-category font-weight-bold">Lokale
                            Informatie</p>
                    </div>
                </CardHeader>
                <CardBody>
                    <Table>
                        <Row>
                            <Col>
                                <Card>
                                    <CardHeader>
                                        <div
                                            className="numbers text-center float-left">
                                            <p className="card-title font-weight-bold">Zie
                                                de inputfeatures van de patient
                                                in</p>
                                        </div>
                                    </CardHeader>
                                    <CardBody>
                                        <Link to={'/admin/dashboard'}>
                                            <Button color={"primary"}>
                                                <div>Input Features</div>
                                                <div
                                                    className={"nc-icon nc-zoom-split"}></div>
                                            </Button>
                                        </Link>
                                    </CardBody>
                                </Card>
                            </Col>
                            <Col>
                                <Card>
                                    <CardHeader>
                                        <div
                                            className="numbers text-center float-left">
                                            <p className="card-title font-weight-bold">Verander
                                                de features van de patient</p>
                                        </div>
                                    </CardHeader>
                                    <CardBody>
                                        <Link to={'/admin/changevalues'}>
                                            <Button color={"primary"}>
                                                <div>Change Values</div>
                                                <div
                                                    className={"nc-icon nc-settings"}></div>
                                            </Button>
                                        </Link>
                                    </CardBody>
                                </Card>
                            </Col>
                        </Row>
                        <Row>
                            <Col>
                                <Card>
                                    <CardHeader>
                                        <div
                                            className="numbers text-center float-left">
                                            <p className="card-title font-weight-bold">Bekijk
                                                de counterfactuals van de
                                                patient</p>
                                        </div>
                                    </CardHeader>
                                    <CardBody>
                                        <Link to={'/admin/counterfactual'}>
                                            <Button color={"primary"}>
                                                <div>Counterfactuals</div>
                                                <div
                                                    className={"nc-icon nc-refresh-69"}></div>
                                            </Button>
                                        </Link>
                                    </CardBody>
                                </Card>
                            </Col>
                            <Col>
                                <Card>
                                    <CardHeader>
                                        <div
                                            className="numbers text-center float-left">
                                            <p className="card-title font-weight-bold">Bekijk
                                                vergelijkbare patienten</p>
                                        </div>
                                    </CardHeader>
                                    <CardBody>
                                        <Link to={'/admin/similar_patients'}>
                                            <Button color={"primary"}>
                                                <div>Similar Patients</div>
                                                <div
                                                    className={"nc-icon nc-single-02"}></div>
                                                <div
                                                    className={"nc-icon nc-single-02"}></div>
                                            </Button>
                                        </Link>
                                    </CardBody>
                                </Card>
                            </Col>
                        </Row>
                    </Table>
                </CardBody>
            </Card>
            <Card>
                <CardBody>
                    <Link to={"/admin/patients"}>
                        <Button>Laad een andere patient in</Button>
                    </Link>
                </CardBody>
            </Card>

        </div>

    );
}

export default Teaser;