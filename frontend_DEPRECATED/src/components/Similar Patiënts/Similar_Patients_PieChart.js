import React from "react";
import {PieChart} from '@mui/x-charts/PieChart';

class Similar_Patients_PieChart extends React.Component {
    valueNee = 10;
    valueJa = 15;
    percentageNee = this.valueNee / (this.valueNee + this.valueJa) * 100;
    percentageJa = this.valueJa / (this.valueNee + this.valueJa) * 100;

    render() {
        return (
            <div className="pie-chart">
                <PieChart
                    series={[
                        {
                            arcLabel: (item) => `${item.label} ${item.value}%`,
                            data: [
                                {
                                    id: 0,
                                    value: this.percentageNee,
                                    label: 'Nee'
                                },
                                {id: 1, value: this.percentageJa, label: 'Ja'},
                            ],
                        },
                    ]}
                    width={400}
                    height={200}
                />
            </div>
        )
    }
}

export default Similar_Patients_PieChart;