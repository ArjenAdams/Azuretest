import React from "react";
import Plot from "react-plotly.js";
import {Card, CardBody} from "reactstrap";

class Heatmap extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: [
                {
                    x: this.props.xValues,
                    y: this.props.yValues,
                    z: this.props.zValues,
                    type: "heatmap",
                    colorscale: this.props.colorscale,
                    showscale: false,
                },
            ],
            layout: {
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
                annotations: this.generateAnnotations(),
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

    generateAnnotations() {
        const annotations = [];
        const {xValues, yValues, zValues} = this.props;

        for (let i = 0; i < yValues.length; i++) {
            for (let j = 0; j < xValues.length; j++) {
                const currentValue = zValues[i][j];
                const textColor = currentValue !== 0.0 ? "white" : "black";

                const annotation = {
                    xref: "x1",
                    yref: "y1",
                    x: xValues[j],
                    y: yValues[i],
                    text: zValues[i][j],
                    font: {
                        family: "Arial",
                        size: 24,
                        color: textColor,
                    },
                    showarrow: false,
                };

                annotations.push(annotation);
            }
        }

        return annotations;
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

export default Heatmap;
