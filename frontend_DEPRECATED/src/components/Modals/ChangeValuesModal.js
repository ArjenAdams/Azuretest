import {Button, Col, Modal, ModalBody, ModalHeader, Row} from "reactstrap";
import React from "react";

function ChangeValuesModal({
                               className,
                               toggle,
                               makePrediction,
                               isOpen,
                               feature
                           }) {
    const body = 'De waarde is aangepast! Nieuwe voorspelling maken?'

    return (
        <div>
            <Modal isOpen={isOpen} toggle={toggle} className={className}>
                <ModalHeader toggle={toggle}>
                    <Row>
                        <Col md={10}
                             className={"align-content-start font-weight-bold "}>
                                 <span className={"align-content-start"}>
                                     Waarde {feature} aangepast:
                                 </span>
                        </Col>
                    </Row>
                </ModalHeader>

                <ModalBody>
                    <p className={"size1rem"}>
                        {body}
                    </p>
                    <Button color="primary"
                            onClick={makePrediction}>Ja</Button>
                    <Button color="secondary" onClick={toggle}>Nog
                        Niet</Button>
                </ModalBody>
            </Modal>
        </div>
    );
}

export default ChangeValuesModal;