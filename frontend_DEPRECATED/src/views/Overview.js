import React from "react";
import Histogram from "../components/Charts/Histogram";
import LineChart from "../components/Charts/LineChart";
import Heatmap from "../components/Charts/Heatmap";
import {Col, Container, Row} from "reactstrap";
import CardComponent from "../components/InformationCards/CardComponent";

function generateRandomData(length, min, max) {
    const data = [];
    for (let i = 0; i < length; i++) {
        data.push(Math.floor(Math.random() * (max - min + 1)) + min);
    }
    return data;
}


function Overview() {
    // Histogram data and properties
    const histogramData = generateRandomData(100, 0, 500)
    const histogramTitle = "Histogram";
    const histogramXAxisTitle = "Value";
    const histogramYAxisTitle = "Count";

    // Bar chart data and properties
    const barXData = ["Category A", "Category B", "Category C", "Category D"];
    const barYData = [12, 20, 15, 23];
    const barTitle = "Bar Chart";
    const barXAxisTitle = "Category";
    const barYAxisTitle = "Value";

    const xValues = ["A", "B", "C", "D", "E"];
    const yValues = ["W", "X", "Y", "Z"];
    const zValues = [
        [0.0, 0.0, 0.75, 0.75, 0.0],
        [0.0, 0.0, 0.75, 0.75, 0.0],
        [0.75, 0.75, 0.75, 0.75, 0.75],
        [0.0, 0.0, 0.0, 0.75, 0.0],
    ];
    const colorscaleValue = [
        [0, "#855CF8"],
        [1, "#263238"],
    ];


    // Line chart data and properties
    const lineXData = ["2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10"];
    const lineYData = [22, 27, 24, 29, 32];
    const lineTitle = "Temperature Trends";
    const lineXAxisTitle = "Date";
    const lineYAxisTitle = "Temperature (°C)";

    const cardData = [
        {title: 'INFO-A', content: 'X', size: 1.5},
        {title: 'INFO-B', content: 'X', size: 1.5},
        {title: 'INFO-C', content: 'X', size: 1.5},
        {title: 'INFO-D', content: 'X', size: 1.5},
    ];

    return (
        <>
            <div className="scaled-content-50">
                <Container fluid={true}>
                    <CardComponent data={cardData}/>
                </Container>

                <Row md={12}>
                    <Col md={12}>
                        <LineChart
                            xData={lineXData}
                            yData={lineYData}
                            title={lineTitle}
                            xAxisTitle={lineXAxisTitle}
                            yAxisTitle={lineYAxisTitle}
                        />
                    </Col>
                </Row>
                <Row>
                    <Col md={6}>
                        <Heatmap
                            xValues={xValues}
                            yValues={yValues}
                            zValues={zValues}
                            colorscale={colorscaleValue}
                            title="Annotated Heatmap"
                            xAxisTitle="X Axis"
                            yAxisTitle="Y Axis"
                        />
                    </Col>
                    <Col md={6}>
                        <Histogram
                            data={histogramData}
                            title={histogramTitle}
                            xAxisTitle={histogramXAxisTitle}
                            yAxisTitle={histogramYAxisTitle}
                            colorscale={colorscaleValue}
                        />
                    </Col>
                </Row>
            </div>
        </>
    );
}

export default Overview;
