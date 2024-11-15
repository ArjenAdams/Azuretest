import React from 'react'
import {Card, CardBody, CardHeader, CardSubtitle, CardTitle} from "reactstrap";
import Plot from "react-plotly.js"; // import the BarChart component


function ImportantFeatures({features}) {

    function getImportantFeatureNamesAndValues() {
        let names = features.map(tuple => tuple[0]);
        let values = features.map(tuple => tuple[1]);
        values = values.map((value => {
            return (value * 100).toFixed(2)
        }))

        return [names, values];
    }

    return (
        <>
            <Card>
                <CardHeader>
                    <CardTitle className="text-dark font-weight-bold text-2rem"
                               tag={"h1"}>
                        Important features
                    </CardTitle>
                    <CardSubtitle className="text-muted size1-25rem"
                                  tag={"h2"}>
                        <span> These are the 5 most influential features for the AI model (percentage based) </span>
                    </CardSubtitle>
                </CardHeader>
                <CardBody>
                    <Plot data={[{
                        type: 'bar',
                        x: getImportantFeatureNamesAndValues()[1], // The values
                        y: getImportantFeatureNamesAndValues()[0], // The feature names
                        orientation: 'h',
                        text: getImportantFeatureNamesAndValues()[1].map(String), // Shows values directly on top of bar chart
                        transforms: [{  // Sorts it from most important to least important
                            type: 'sort',
                            target: 'x',
                            order: 'ascending'
                        }]
                    }]}
                          layout={{
                              automargin: true,
                              yaxis: {
                                  title: 'Feature',
                                  titlefont: {
                                      color: 'lightgrey'
                                  },
                                  automargin: true
                              },
                              xaxis: {
                                  title: 'Value',
                                  titlefont: {
                                      color: 'lightgrey'
                                  },
                                  fontStyle: 'bold',
                              }

                          }}></Plot>
                </CardBody>
            </Card>

        </>
    )
}

export default ImportantFeatures
