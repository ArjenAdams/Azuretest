import React from "react";
import Plot from "react-plotly.js";
import {
    Card,
    CardBody,
    CardFooter,
    CardHeader,
    CardTitle,
    Col,
    Container,
    Row,
} from "reactstrap";

class LineChart extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: [
                {
                    x: this.props.xData,
                    y: this.props.yData,
                    type: "scatter",
                    mode: "lines",
                    line: {color: "rgb(0, 107, 164)"},
                },
            ],
            layout: {
                autosize: true,
                showlegend: false,
                xaxis: {
                    ticks: "",
                    tickfont: {
                        size: 20,
                        padded: true,
                        padding: 10,
                    },
                },
                yaxis: {
                    title: {
                        text: this.props.yAxisTitle,
                        font: {
                            size: 36,

                        }
                    },
                    ticks: "",
                    ticksuffix: " ",
                    tickfont: {
                        size: 20,
                    },
                },
            },
        };
    }

    render() {
        return (
            <Container fluid={true}>
                <Card>
                    <CardHeader>
                        <Row>
                            <Col>
                                <h2 className="text-center font-weight-bold">
                                    {this.props.title}
                                </h2>
                            </Col>
                        </Row>
                    </CardHeader>
                    <CardBody>
                        <Row>
                            <Col md={2}>
                                {this.props.yData.map((value, index) => {
                                    return (
                                        <Row key={index}>
                                            <Col md={12}>
                                                <Card>
                                                    <CardHeader>
                                                        <Row>
                                                            <Col>
                                                                <div
                                                                    className="numbers text-center text-center">
                                                                    <p className="card-category font-weight-bold">
                                                                        {value}
                                                                    </p>
                                                                </div>
                                                            </Col>
                                                        </Row>
                                                    </CardHeader>
                                                    <CardBody>
                                                        <Row>
                                                            <Col>
                                                                <div
                                                                    className="numbers text-center">
                                                                    <CardTitle
                                                                        className={`font-weight-bold size1-5rem`}>
                                                                        <p className={"font-weight-bolder"}>
                                                                            {this.props.xData[index]}
                                                                        </p>
                                                                    </CardTitle>
                                                                </div>
                                                            </Col>
                                                        </Row>
                                                    </CardBody>
                                                </Card>
                                            </Col>
                                        </Row>
                                    );
                                })}
                            </Col>
                            <Col md={10}>
                                <Plot
                                    data={this.state.data}
                                    layout={this.state.layout}
                                    useResizeHandler={true}
                                    style={{width: "100%", height: "100%"}}
                                />
                            </Col>
                        </Row>
                    </CardBody>
                    <CardFooter>
                        <Row>
                            <Col>
                                <h4 className="text-center font-weight-bold">
                                    {this.props.xAxisTitle}
                                </h4>
                            </Col>
                        </Row>
                    </CardFooter>
                </Card>
            </Container>
        );
    }
}

export default LineChart;
