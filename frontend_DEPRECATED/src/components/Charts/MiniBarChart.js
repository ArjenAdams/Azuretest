import React from 'react';
import Plot from 'react-plotly.js';

const MiniBarChart = ({xData, yData}) => {

    const data = [
        {
            type: 'bar',
            x: xData,
            y: yData,
            orientation: 'h',
            marker: {
                color: 'rgb(0, 107, 164)'
            }
        }
    ];

    const layout = {
        autosize: true,
        margin: {
            l: 20,
            r: 5,
            b: 5,
            t: 5,
            pad: 4,
        },
        xaxis: {
            range: [0, 1],  // set the range of x-axis
            ticks: '',
            side: 'bottom',
            showticklabels: false,
            showgrid: false,
        },
        yaxis: {
            ticks: '',
            ticksuffix: ' ',
            showticklabels: false,
            showgrid: false,
        },
        showlegend: false,
        bargap: 0.05, // gap between bars of adjacent location coordinates
        bargroupgap: 0.1, // gap between bars of the same location coordinate
        height: 50, // adjust based on your preference
        annotations: [
            {
                xref: 'paper',
                yref: 'paper',
                x: 0,
                xanchor: 'left',
                y: 0.5,
                yanchor: 'middle',
                text: yData[0].toString(), // Make sure the value is converted to string
                font: {
                    color: 'black',  // Change text color for contrast
                    size: 12         // Adjust text size as needed
                },
                showarrow: false
            }
        ]
    };

    return (
        <Plot data={data} layout={layout}
              style={{width: "50%", height: "100%"}}
              config={{displayModeBar: false}}/>
    );
};

export default MiniBarChart;
