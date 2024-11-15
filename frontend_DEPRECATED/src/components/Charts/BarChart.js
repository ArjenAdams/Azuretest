import React from "react";
import Plot from "react-plotly.js";
import {Card, CardBody} from "reactstrap";

class BarChart extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: [
                {
                    x: this.props.xData,
                    y: this.props.yData,
                    type: "bar",
                    marker: {
                        color: "rgb(0, 107, 164)",
                    },
                },
            ],
            layout: {
                autosize: true,
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

export default BarChart;
