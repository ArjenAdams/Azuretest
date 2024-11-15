import React from 'react'
import {Progress} from 'reactstrap';

function Similar_Patients_Bar(data) {
    console.log(data)
    // Methode 1, bar gebruiken met een predictie
    let valueA;
    let valueB;
    switch (data.method) {
        case("hoeveelheidPatientenMetSAD"):
            valueA = (data.aantalPatientenMetSAD / data.aantalPatienten * 100).toFixed(2)
            valueB = ((data.aantalPatienten - data.aantalPatientenMetSAD) / data.aantalPatienten * 100).toFixed(2)
            break;
        case("predictieWeergeven"):
            valueA = Math.round(data.prediction).toFixed(2);
            valueB = 100 - Math.round(data.prediction).toFixed(2);
            break;
        default:
            valueA = Math.round(data.prediction).toFixed(2);
            valueB = 100 - Math.round(data.prediction).toFixed(2);
            break;
    }


    return (
        <div className="bar-chart">
            <Progress multi>
                <Progress bar color="success"
                          value={valueA}>{valueA}%</Progress>
                <Progress bar color="danger"
                          value={valueB}>{valueB}%</Progress>
            </Progress>
        </div>
    )
}

export default Similar_Patients_Bar