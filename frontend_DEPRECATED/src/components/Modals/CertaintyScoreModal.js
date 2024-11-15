import {
    Button,
    Col,
    Modal,
    ModalBody,
    ModalFooter,
    ModalHeader,
    Row
} from "reactstrap";
import React, {useState} from "react";

function CertaintyScoreModal({score, className}) {
    const certaintyScore = score;
    const [isOpen, setIsOpen] = useState(false);
    const body = <p>
        Wat de uitkomst van het model betekent is hoe zeker het model is dat
        iemand <strong>wel</strong> SAD heeft,
        de reciprocal hiervan is dus hoe zeker iemand <strong>niet</strong> SAD
        heeft.
        <br></br>
        <br></br>
        Hoe je zekerheid dus afleest:
        <br></br>
        - Stel het model zegt 0.68, dan is die 68% zeker dat het wel SAD is en
        32% zeker dat het niet SAD is.
        De voorspelling is dan `True` en de zekerheidscore is dan 68%.
        <br></br>
        - Stel het model zegt 0.11, dan is die 11% zeker dat het wel SAD is en
        89% zeker van niet.
        De voorspelling is dan `False` en de zekerheid is dan 89%.
    </p>


    function toggle() {
        setIsOpen(!isOpen);
    }

    return (
        <div>
            <Button className={"btn-primary"} onClick={toggle}>
                {
                    <i className="nc-icon nc-tap-01"/>
                }
            </Button>

            <Modal isOpen={isOpen} toggle={toggle} className={className}>
                <ModalHeader toggle={toggle}>
                    <Row>
                        <Col md={10}
                             className={"align-content-start font-weight-bold "}>
                                 <span className={"align-content-start"}>
                                     Certainty score:
                                 </span>
                        </Col>
                        <Col md={2}
                             className={"align-content-end font-weight-bold size2rem text-blue"}>
                                <span>
                                    {certaintyScore}
                                </span>
                        </Col>
                    </Row>
                </ModalHeader>

                <ModalBody>
                    <p className={"size1rem"}>
                        {body}
                    </p>
                </ModalBody>

                <ModalFooter>
                    <Button color="secondary" onClick={toggle}>Hide</Button>
                </ModalFooter>
            </Modal>
        </div>
    );
}

export default CertaintyScoreModal;