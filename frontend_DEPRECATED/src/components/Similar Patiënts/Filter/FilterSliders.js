import React from "react";

import {
    createTheme,
    makeStyles,
    MuiThemeProvider
} from "@material-ui/core/styles";
import Typography from "@material-ui/core/Typography";
import Slider from "@material-ui/core/Slider";


const useStyles = makeStyles({
    root: {
        width: "60%"
    }
});
const numberMarks = [
    {
        value: 0,
        label: "0"
    },
    {
        value: 100,
        label: "100"
    }
];


const theme = createTheme({
    palette: {
        primary: {main: "rgb(38, 118, 198)"}
    },
    typography: {useNextVariants: true}
});

export default function RangeSlider() {

    const classes = useStyles();

    const [value, setValue] = React.useState([0, 100]);
    const [BMIvalue, setBMIValue] = React.useState([0, 100]);
    const [operatieValue, setOperatieValue] = React.useState([0, 100]);
    const [asaValue, setAsaValue] = React.useState([0, 100]);
    const [verblijfsduurValue, setVerblijfsduurValue] = React.useState([0, 100]);
    const [medicijnenValue, setMedicijnenValue] = React.useState([0, 100]);
    const [diagnosesValue, setDiagnosesValue] = React.useState([0, 100]);

    const handleChange = (event, newValue) => {
        setValue(newValue);
    };
    const handleBMIChange = (event, newValue) => {
        setBMIValue(newValue);
    };
    const handleOperatieChange = (event, newValue) => {
        setOperatieValue(newValue);
    };
    const handleAsaChange = (event, newValue) => {
        setAsaValue(newValue);
    }
    const handleVerblijfsduurChange = (event, newValue) => {
        setVerblijfsduurValue(newValue);
    }
    const handleMedicijnenChange = (event, newValue) => {
        setMedicijnenValue(newValue);
    }
    const handleDiagnosesChange = (event, newValue) => {
        setDiagnosesValue(newValue);
    }


    return (
        <MuiThemeProvider theme={theme}>
            <div className={classes.root}>
                <Typography id="range-slider" gutterBottom>
                    Leeftijd
                </Typography>
                <Slider
                    aria-labelledby="range-slider"
                    color="primary"
                    defaultValue={50}
                    onChange={handleChange}
                    marks={numberMarks}
                    value={value}
                    valueLabelDisplay="auto"
                />

                <Typography id="range-slider" gutterBottom>
                    BMI
                </Typography>
                <Slider
                    aria-labelledby="range-slider"
                    color="primary"
                    defaultValue={50}
                    onChange={handleBMIChange}
                    marks={numberMarks}
                    value={BMIvalue}
                    valueLabelDisplay="auto"
                />

                <Typography id="range-slider" gutterBottom>
                    Tijdsduur in operatiekamer (minuten)
                </Typography>
                <Slider
                    aria-labelledby="range-slider"
                    color="primary"
                    defaultValue={50}
                    onChange={handleOperatieChange}
                    marks={numberMarks}
                    value={operatieValue}
                    valueLabelDisplay="auto"
                />

                <Typography id="range-slider" gutterBottom>
                    ASA score
                </Typography>
                <Slider
                    aria-labelledby="range-slider"
                    color="primary"
                    defaultValue={50}
                    onChange={handleAsaChange}
                    marks={numberMarks}
                    value={asaValue}
                    valueLabelDisplay="auto"
                />

                <Typography id="range-slider" gutterBottom>
                    Verblijfsduur tot operatie (uren)
                </Typography>
                <Slider
                    aria-labelledby="range-slider"
                    color="primary"
                    defaultValue={50}
                    onChange={handleVerblijfsduurChange}
                    marks={numberMarks}
                    value={verblijfsduurValue}
                    valueLabelDisplay="auto"
                />

                <Typography id="range-slider" gutterBottom>
                    Aantal unieke medicijnen
                </Typography>
                <Slider
                    aria-labelledby="range-slider"
                    color="primary"
                    defaultValue={50}
                    onChange={handleMedicijnenChange}
                    marks={numberMarks}
                    value={medicijnenValue}
                    valueLabelDisplay="auto"
                />

                <Typography id="range-slider" gutterBottom>
                    Aantal diagnoses
                </Typography>
                <Slider
                    aria-labelledby="range-slider"
                    color="primary"
                    defaultValue={50}
                    onChange={handleDiagnosesChange}
                    marks={numberMarks}
                    value={diagnosesValue}
                    valueLabelDisplay="auto"
                />
            </div>
        </MuiThemeProvider>
    );
}