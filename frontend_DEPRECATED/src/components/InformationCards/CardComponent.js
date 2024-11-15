import React, {Component} from 'react';
import {Card, CardBody, CardHeader, CardTitle, Col, Row} from 'reactstrap';

class CardComponent extends Component {
    renderCards = () => {
        const {data} = this.props;
        const colSize = 12 / data.length;

        return (
            <Row>
                {data.map((card, index) => (
                    <Col md={colSize} key={index}>
                        <Card>
                            <CardHeader>
                                <Row>
                                    <Col>
                                        <div
                                            className="numbers text-center text-center">
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
                    </Col>
                ))}
            </Row>
        );
    };

    render() {
        return <>{this.renderCards()}</>;
    }
}

export default CardComponent;
