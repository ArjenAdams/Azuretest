import React, {Component} from 'react';
import {Card, CardBody, CardHeader, CardTitle, Col, Row} from 'reactstrap';

class CustomCard extends Component {
    render() {
        const {card} = this.props;
        return (
            <Card>
                <CardHeader>
                    <Row>
                        <Col>
                            <div className="numbers text-center">
                                <p className="card-category font-weight-bold">{card.title}</p>
                            </div>
                        </Col>
                    </Row>
                </CardHeader>
                <CardBody>
                    <Row>
                        <Col>
                            <div className="numbers text-center">
                                <CardTitle
                                    className={`font-weight-bold size${card.size}-rem`}>
                                    <p className={"font-weight-bolder"}>
                                        {card.content}
                                    </p>
                                </CardTitle>
                            </div>
                        </Col>
                    </Row>
                </CardBody>
            </Card>
        );
    }
}

export default CustomCard;
