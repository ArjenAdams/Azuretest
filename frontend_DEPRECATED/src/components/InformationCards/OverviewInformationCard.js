import React from "react";
import {Card, CardBody, CardHeader, CardTitle, Col, Row} from "reactstrap";
import InformationModal from "../Modals/InformationModal";


function OverviewInformationCard(props) {
    const {cardInformation} = props;
    console.log(cardInformation);


    return (
        <>
            <Row>
                <Col lg="4" md="6" sm="6" className={"h-100"}>
                    <Card className={"h-100"}>
                        <CardHeader>
                            <Row>
                                <Col>
                                    <div
                                        className="numbers text-center float-left">
                                        <p className="card-category font-weight-bold">Patient
                                            ID</p>
                                    </div>
                                </Col>
                            </Row>
                        </CardHeader>
                        <CardBody>
                            <Row>
                                <Col>
                                    <div className="numbers text-center">
                                        <CardTitle
                                            className="font-weight-bold size1-5rem">
                                            <p className={"font-weight-bolder"}>
                                                {cardInformation.patientId}
                                            </p>
                                        </CardTitle>
                                    </div>
                                </Col>
                            </Row>
                        </CardBody>
                    </Card>
                </Col>

                <Col lg="4" md="6" sm="6" className={"h-100"}>
                    <Card className={"h-100"}>
                        <CardHeader>
                            <Row>
                                <Col>
                                    <div className="numbers text-center">
                                        <p className="card-category text-blue font-weight-bolder">
                                            Model prediction
                                        </p>
                                    </div>
                                </Col>
                            </Row>
                        </CardHeader>
                        <CardBody>
                            <Row>
                                <Col>
                                    <div className="numbers text-center">
                                        <CardTitle
                                            className="font-italic size1-5rem font-weight-bold">
                                            {cardInformation.modelPrediction}
                                        </CardTitle>
                                    </div>
                                </Col>
                            </Row>
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
                                    <InformationModal/>
                                </Col>
                            </Row>
                        </CardHeader>
                        <CardBody>
                            <Row>
                                <Col>
                                    <div className="numbers text-center">
                                        <CardTitle
                                            className="font-italic size1-25rem">
                                            <p className={"font-weight-bolder size1-5rem"}>
                                                {cardInformation.certaintyScore}
                                            </p>
                                        </CardTitle>
                                    </div>
                                </Col>
                            </Row>
                        </CardBody>
                    </Card>
                </Col>
            </Row>

        </>
    )
}

export default OverviewInformationCard;
