import {
    Button,
    Col,
    Modal,
    ModalBody,
    ModalFooter,
    ModalHeader,
    Row
} from "reactstrap";
import React from "react";

function PredictionChangedModal({
                                    className,
                                    toggle,
                                    isOpen,
                                    previousPrediction,
                                    newPrediction,
                                    newCertaintyScore,
                                    previousCertaintyScore,
                                    features
                                }) {

    return (
        <div>
            <Modal isOpen={isOpen} toggle={toggle} className={className}>
                <ModalHeader toggle={toggle}>
                    <Row>
                        <Col md={10}
                             className={"align-content-start font-weight-bold "}>
                                 <span className={"align-content-start"}>
                                     De voorspelling is aangepast:
                                 </span>
                        </Col>
                    </Row>
                </ModalHeader>

                <ModalBody>
                    <p className={"size1rem"}>
                        Heeft de patient SAD?
                        <br/>
                        Vorige voorspelling: {previousPrediction}
                        <br/>
                        Nieuwe voorspelling: {newPrediction}
                        <br/>
                        <br/>
                        Vorige zekerheidscore: {previousCertaintyScore}
                        <br/>
                        Nieuwe zekerheidscore: {newCertaintyScore}
                    </p>
                    <br/>
                    <br/>
                    <p className={"size1rem"}>
                        Veranderde waarde(s):
                    </p>

                    <div>
                        {features.map((feature, index) => (
                            <p key={index}>
                                {feature}
                            </p>
                        ))}
                    </div>
                </ModalBody>
                <ModalFooter>
                    <Button color="secondary" onClick={toggle}>Hide</Button>
                </ModalFooter>
            </Modal>
        </div>
    );
}

export default PredictionChangedModal;