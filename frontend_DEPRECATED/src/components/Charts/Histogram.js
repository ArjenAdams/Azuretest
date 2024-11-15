import React from "react";
import Plot from "react-plotly.js";
import {Card, CardBody} from "reactstrap";

class Histogram extends React.Component {
    constructor(props) {
        super(props);

        const data1 = this.generateRandomData(500, 1, 5);
        const data2 = this.generateRandomData(500, 1, 10);

        this.state = {
            data: [
                {
                    x: data1,
                    name: "control",
                    autobinx: false,
                    histnorm: "count",
                    marker: {
                        color: "rgba(255, 100, 102, 0.7)",
                        line: {
                            color: "rgba(255, 100, 102, 1)",
                            width: 1,
                        },
                    },
                    opacity: 0.5,
                    type: "histogram",
                    xbins: {
                        end: 2.8,
                        size: 0.06,
                        start: 0.5,
                    },
                },
                {
                    x: data2,
                    name: "experimental",
                    autobinx: false,
                    marker: {
                        color: "rgba(100, 200, 102, 0.7)",
                        line: {
                            color: "rgba(100, 200, 102, 1)",
                            width: 1,
                        },
                    },
                    opacity: 0.75,
                    type: "histogram",
                    xbins: {
                        end: 4,
                        size: 0.06,
                        start: -3.2,
                    },
                },
            ],
            layout: {
                autosize: true,
                bargap: 0.05,
                bargroupgap: 0.2,
                barmode: "overlay",
                margin: {
                    l: 75,
                    r: 75,
                    b: 75,
                    t: 75,
                    pad: 4,
                },
                title: {
                    text: this.props.title,
                    font: {
                        size: 24,
                    },
                },
                xaxis: {
                    ticks: "",
                    side: "bottom",
                    title: {
                        text: this.props.xAxisTitle,
                        font: {
                            size: 36,
                        },
                    },
                    tickfont: {
                        size: 20,
                    },
                },
                yaxis: {
                    ticks: "",
                    ticksuffix: " ",
                    title: {
                        text: this.props.yAxisTitle,
                        font: {
                            size: 36,
                        },
                    },
                    tickfont: {
                        size: 20,
                    },
                },
            },
        };
    }

    generateRandomData(length, min, max) {
        const data = [];
        for (let i = 0; i < length; i++) {
            const k = Math.random();
            data.push(k * (max - min) + min);
        }
        return data;
    }

    render() {
        return (
            <Card>
                <CardBody>
                    <div className="chart">
                        <Plot
                            data={this.state.data}
                            layout={this.state.layout}
                            useResizeHandler={true}
                            style={{width: "100%", height: "100%"}}
                        />
                    </div>
                </CardBody>
            </Card>
        );
    }
}

export default Histogram;
