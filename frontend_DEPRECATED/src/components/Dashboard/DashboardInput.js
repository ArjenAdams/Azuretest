import React from 'react';
import {
    Card,
    CardBody,
    CardHeader,
    CardSubtitle,
    CardTitle,
    Col,
    Label,
    Row,
    Table
} from 'reactstrap';

function DashboardInput(props) {
    const featureKeys = Object.keys(props.features);
    const halfLength = Math.ceil(featureKeys.length / 2);
    const renderFeatureTable = (startIndex, endIndex) => (
        <Col>
            <Table className="text-center">
                <tbody>
                {featureKeys.slice(startIndex, endIndex).map((featureKey, index) => (
                    <tr key={index}>
                        <td>
                            <Label tag={"p"}
                                   className="size1-5rem font-weight-bold text-muted">
                                {
                                    featureKey
                                }
                                <i style={{
                                    justifyContent: "center",
                                    display: "flex",
                                    padding: "0px",
                                    margin: "0px"
                                }}>
                                    <p style={{
                                        fontSize: '60%',
                                        fontWeight: "normal",
                                        textAlign: 'center',
                                        maxWidth: "70%",
                                        padding: "0px",
                                        margin: "0px"
                                    }}>
                                        {props.features[featureKey].description}
                                    </p>
                                </i>
                            </Label>
                        </td>
                        <td>
                            {typeof (props.features[featureKey].value) === "number" && props.features[featureKey].value.toString().includes(".") ? (
                                <Label name={featureKey} id={featureKey}
                                       className="float-left size1-5rem font-weight-bold text-muted"
                                       style={{
                                           width: '100%',
                                           margin: 'auto',
                                           fontSize: "100%"
                                       }}>
                                    {Number.parseFloat(props.features[featureKey].value.toString()).toFixed(2)}</Label>
                            ) : (
                                <Label name={featureKey} id={featureKey}
                                       className="float-left size1-5rem font-weight-bold text-muted"
                                       style={{
                                           width: '100%',
                                           margin: 'auto',
                                           fontSize: "100%"
                                       }}>

                                    {props.features[featureKey].value}</Label>)
                            }
                        </td>
                    </tr>
                ))}
                </tbody>
            </Table>
        </Col>
    );

    return (
        <>
            <Card>
                <CardHeader>
                    <CardTitle className="text-dark font-weight-bold text-2rem"
                               tag={"h1"}>
                        Input features for this decision
                    </CardTitle>
                    <CardSubtitle className="text-muted size1-25rem"
                                  tag={"h2"}>
                        <span> The model only predicts the features with valid values. Blank values are not predicted. </span>
                    </CardSubtitle>
                </CardHeader>
                <CardBody>
                    <Row>
                        {renderFeatureTable(0, halfLength)}
                        {renderFeatureTable(halfLength, featureKeys.length)}
                    </Row>
                </CardBody>
            </Card>
        </>
    );
}

export default DashboardInput;
