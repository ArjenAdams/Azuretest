import {Card, CardBody, CardHeader, CardTitle, Col, Row} from "reactstrap";
import ModelProgress from "../Model/ModelProgress";
import CertaintyScoreModal from "../Modals/CertaintyScoreModal";
import React from "react";

function ModelInfoCard() {
    const certaintyScore = 96;
    const cardInformation = {"Patient ID": "123456789"};
    const predictionResult = "Ja";

    return (
        <Row>
            <Col lg="4" md="6" sm="6">
                <Card>
                    <CardHeader>
                        <div className="numbers">
                            <CardTitle
                                className="card-category font-weight-bold">Patiënt</CardTitle>
                        </div>
                    </CardHeader>
                    <CardBody>
                        <div className="numbers text-center">
                            <CardTitle className="font-weight-bold size1-5rem">
                                <p className={"font-weight-bolder"}>
                                    {cardInformation["Patient ID"] || ""}
                                </p>
                            </CardTitle>
                        </div>
                    </CardBody>
                </Card>
            </Col>
            <Col lg="4" md="6" sm="6">
                <Card>
                    <CardHeader>
                        <div className="numbers">
                            <CardTitle
                                className="card-category font-weight-bold">Veilig
                                ontslag uit ziekenhuis?</CardTitle>
                        </div>
                    </CardHeader>
                    <CardBody>
                        <div className="numbers text-center">
                            <CardTitle
                                className="font-italic size1-5rem font-weight-bold">
                                {predictionResult}
                            </CardTitle>
                        </div>
                    </CardBody>
                </Card>
            </Col>

            <Col lg="4" md="6" sm="6">
                <Card>
                    <CardHeader>
                        <Row md={12}>
                            <Col md={9}>
                                <div className="numbers">
                                    <CardTitle
                                        className="card-category font-weight-bold">Certainty
                                        Score</CardTitle>
                                </div>
                            </Col>
                            <Col md={3}>
                                <CertaintyScoreModal score={certaintyScore}/>
                            </Col>
                        </Row>
                    </CardHeader>
                    <CardBody>
                        <ModelProgress percentage={certaintyScore}/>
                    </CardBody>
                </Card>
            </Col>
        </Row>
    );
}

export default ModelInfoCard;