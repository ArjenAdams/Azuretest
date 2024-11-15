import React from "react";
import {Card, CardBody, CardHeader, CardTitle, Col, Row} from "reactstrap";
import CertaintyScoreModal from "../Modals/CertaintyScoreModal";

function DashboardInfoCard({cardInformation}) {
    const translatePrediction = (prediction) => {
        if (Number(prediction) < 0.5) {
            return "Nee"
        } else if (Number(prediction) >= 0.5) {
            return "Ja"
        } else {
            return "Unknown"
        }
    }

    return (
        <Row>
            <Col lg="4" md="6" sm="6" className={"h-100"}>
                <Card className={"h-100"}>
                    <CardHeader>
                        <div className="numbers text-center float-left">
                            <p className="card-category font-weight-bold">Patient
                                ID</p>
                        </div>
                    </CardHeader>
                    <CardBody>
                        <div className="numbers text-center">
                            <CardTitle className="font-weight-bold size1-5rem">
                                <p className={"font-weight-bolder"}>
                                    {cardInformation["Patient ID"] || "Unknown"}
                                </p>
                            </CardTitle>
                        </div>
                    </CardBody>
                </Card>
            </Col>

            <Col lg="4" md="6" sm="6" className={"h-100"}>
                <Card className={"h-100"}>
                    <CardHeader>
                        <div className="numbers text-center">
                            <p className="card-category text-blue font-weight-bolder">
                                Heeft SAD?
                            </p>
                        </div>
                    </CardHeader>
                    <CardBody>
                        <div className="numbers text-center">
                            <CardTitle
                                className="font-italic size1-5rem font-weight-bold">
                                {translatePrediction(cardInformation["Prediction"])}
                            </CardTitle>
                        </div>
                    </CardBody>
                </Card>
            </Col>

            <Col lg="4" md="6" sm="6" className={"h-100"}>
                <Card>
                    <CardHeader>
                        <Row md={12}>
                            <Col md={9}>
                                <div className="numbers">
                                    <p className="card-category font-weight-bold align-content-center">Certainty
                                        score</p>
                                </div>
                            </Col>
                            <Col md={3}>
                                <CertaintyScoreModal
                                    score={cardInformation["Probability"]}/>
                            </Col>
                        </Row>
                    </CardHeader>
                    <CardBody>
                        <div className="numbers text-center">
                            <CardTitle className="font-italic size1-25rem">
                                <p className={"font-weight-bolder size1-5rem"}>
                                    {cardInformation["Probability"] || "Unknown"}
                                </p>
                            </CardTitle>
                        </div>
                    </CardBody>
                </Card>
            </Col>
        </Row>
    );
}

export default DashboardInfoCard;
