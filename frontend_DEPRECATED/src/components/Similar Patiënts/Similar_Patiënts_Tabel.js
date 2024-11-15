import {
    Button,
    Card,
    CardBody,
    CardHeader,
    CardSubtitle,
    CardTitle,
    Col,
    Pagination,
    PaginationItem,
    PaginationLink,
    Row,
} from 'reactstrap';

import * as React from 'react';
import {useState} from 'react';
import Box from '@mui/material/Box';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import FilterBox from "./Filter/FilterBox";
import Similar_Patients_Bar from "./Similar_Patients_Bar";
import AdvancedFilter from "./Filter/AdvancedFilter";

let ownPatient = {}

function createData(patientData) {
    let gender = "F"
    // Gender converten naar getallen
    if (patientData.gender === 1) {
        gender = "M"
    }
    let ownGender = "F"
    // Gender converten naar getallen
    if (ownPatient.gender === 1) {
        ownGender = "M"
    }
    return {
        info: [
            {
                "age": patientData.age,
                "weight": patientData.weight,
                "gender": gender,
                "temperature": patientData.temperature,
                "heart_rate": patientData.heart_rate,
                "resp_rate": patientData.resp_rate,
                "spo2": patientData.spo2,
                "sbp": patientData.sbp,
                "dbp": patientData.dbp,
                "mbp": patientData.mbp,
                "wbc": patientData.wbc,
                "hemoglobin": patientData.hemoglobin,
                "platelet": patientData.platelet,
                "bun": patientData.bun,
                "cr": patientData.cr,
                "glu": patientData.glu,
                "Na": patientData.Na,
                "Cl": patientData.Cl,
                "K": patientData.K,
                "Mg": patientData.Mg,
                "Ca": patientData.Ca,
                "P": patientData.P,
                "inr": patientData.inr,
                "pt": patientData.pt,
                "ptt": patientData.ptt,
                "bicarbonate": patientData.bicarbonate,
                "aniongap": patientData.aniongap,
                "gcs": patientData.gcs,
                "vent": patientData.vent,
                "crrt": patientData.crrt,
                "vaso": patientData.vaso,
                "seda": patientData.seda,
                "sofa_score": patientData.sofa_score,
                "ami": patientData.ami,
                "ckd": patientData.ckd,
                "copd": patientData.copd,
                "hyperte": patientData.hyperte,
                "dm": patientData.dm,
                "aki": patientData.aki,
                "stroke": patientData.stroke,
                "AISAN": patientData.AISAN,
                "BLACK": patientData.BLACK,
                "HISPANIC": patientData.HISPANIC,
                "OTHER": patientData.OTHER,
                "WHITE": patientData.WHITE,
                "unknown": patientData.unknown,
                "CCU": patientData.CCU,
                "CVICU": patientData.CVICU,
                "MICU": patientData.MICU,
                "MICU/SICU": patientData["MICU/SICU"],
                "NICU": patientData.NICU,
                "SICU": patientData.SICU,
                "TSICU": patientData.TSICU
            },
            {
                "age": ownPatient.age,
                "weight": ownPatient.weight,
                "gender": ownGender,
                "temperature": ownPatient.temperature,
                "heart_rate": ownPatient.heart_rate,
                "resp_rate": ownPatient.resp_rate,
                "spo2": ownPatient.spo2,
                "sbp": ownPatient.sbp,
                "dbp": ownPatient.dbp,
                "mbp": ownPatient.mbp,
                "wbc": ownPatient.wbc,
                "hemoglobin": ownPatient.hemoglobin,
                "platelet": ownPatient.platelet,
                "bun": ownPatient.bun,
                "cr": ownPatient.cr,
                "glu": ownPatient.glu,
                "Na": ownPatient.Na,
                "Cl": ownPatient.Cl,
                "K": ownPatient.K,
                "Mg": ownPatient.Mg,
                "Ca": ownPatient.Ca,
                "P": ownPatient.P,
                "inr": ownPatient.inr,
                "pt": ownPatient.pt,
                "ptt": ownPatient.ptt,
                "bicarbonate": ownPatient.bicarbonate,
                "aniongap": ownPatient.aniongap,
                "gcs": ownPatient.gcs,
                "vent": ownPatient.vent,
                "crrt": ownPatient.crrt,
                "vaso": ownPatient.vaso,
                "seda": ownPatient.seda,
                "sofa_score": ownPatient.sofa_score,
                "ami": ownPatient.ami,
                "ckd": ownPatient.ckd,
                "copd": ownPatient.copd,
                "hyperte": ownPatient.hyperte,
                "dm": ownPatient.dm,
                "aki": ownPatient.aki,
                "stroke": ownPatient.stroke,
                "AISAN": ownPatient.AISAN,
                "BLACK": ownPatient.BLACK,
                "HISPANIC": ownPatient.HISPANIC,
                "OTHER": ownPatient.OTHER,
                "WHITE": ownPatient.WHITE,
                "unknown": ownPatient.unknown,
                "CCU": ownPatient.CCU,
                "CVICU": ownPatient.CVICU,
                "MICU": ownPatient.MICU,
                "MICU/SICU": ownPatient["MICU/SICU"],
                "NICU": ownPatient.NICU,
                "SICU": ownPatient.SICU,
                "TSICU": ownPatient.TSICU
            }
        ],
        veilig_ontslag: patientData.prediction,
        certaintyscore: patientData.certaintyscore
    }
}

function styleRow(row) {
    //     make first letter uppercase
    row = row.charAt(0).toUpperCase() + row.slice(1);
    // remove underscores
    row = row.replace(/_/g, ' ');
    return row;
}

function translatePrediction(prediction) {
    if (Number(prediction) < 0.5) {
        return "Nee"
    } else if (Number(prediction) >= 0.5) {
        return "Ja"
    } else {
        return "Unknown"

    }
}

function Expanded_Row(details) {
    const [open, setOpen] = React.useState(false);
    const row = createData(details.details);

    let titles = Object.keys(row.info[0]);

    for (let r in row.info[0]){
        if(r.value !== undefined){
            titles = Object.keys(r)
        }
    }


    return (
        <React.Fragment>
            <TableRow sx={{'& > *': {borderBottom: 'unset'}}}>
                <TableCell align="right"
                           className="font-weight-bold">{translatePrediction(details.prediction)}</TableCell>
                <TableCell align="right"
                           className="font-weight-bold">{details.confidence}</TableCell>
                <TableCell align="right">
                    <IconButton
                        aria-label="expand row"
                        size="small"
                        onClick={() => setOpen(!open)}
                    >
                        {open ? <KeyboardArrowUpIcon/> :
                            <KeyboardArrowDownIcon/>}
                    </IconButton>
                </TableCell>
            </TableRow>

            <TableRow>
                <TableCell style={{paddingBottom: 0, paddingTop: 0}}
                           colSpan={6}>
                    <Collapse in={open} timeout="auto" unmountOnExit>
                        <Box sx={{margin: 1}}>
                            <Table size="small" aria-label="purchases">
                                <TableHead>
                                    <TableRow>
                                        <TableCell></TableCell>
                                        <TableCell
                                            className="text-muted font-weight-bold">Soortgelijke
                                            patiënt</TableCell>
                                        <TableCell
                                            className="text-muted font-weight-bold">Uw
                                            patiënt</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {titles.map((infoRow, index) => (
                                        <TableRow key={index}>
                                            <TableCell
                                                className="text-muted font-weight-bold">{styleRow(infoRow)}</TableCell>
                                            {typeof (row.info[0][infoRow]) === "number" &&
                                            row.info[0][infoRow].toString().includes(".") ? (
                                                <TableCell>{row.info[0][infoRow].toFixed(2)}</TableCell>
                                            ) : (
                                                <TableCell>{row.info[0][infoRow]}</TableCell>
                                            )}
                                            {typeof (row.info[1][infoRow]) === "number" &&
                                            row.info[1][infoRow].toString().includes(".") ? (
                                                <TableCell>{row.info[1][infoRow].toFixed(2)}</TableCell>

                                            ) : (
                                                <TableCell>{row.info[1][infoRow]}</TableCell>
                                            )}
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </Box>
                    </Collapse>
                </TableCell>
            </TableRow>
        </React.Fragment>
    );
}

function Similar_Patients_Table(data) {
    console.log("Similar Patients Table")
    let patients = data.similar_patients.patients;
    ownPatient = data.own_patient
    const prediction = data.prediction;
    const predictions = data.similar_patients.predictions;
    const confidence = data.similar_patients.confidence;
    console.log(data.similar_patients)
    const [currentPage, setCurrentPage] = useState(0);
    const pageSize = 10; // Change this to your desired page size
    const pageCount = Math.ceil(patients.length / pageSize);
    const currentPatients = patients.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
    let count = 0;
    for (let pred in predictions) {
        if (predictions[pred] >= 0.5) {
            count += 1
        }
    }
    return (
        <>
            <Card>
                <CardHeader>
                    <CardTitle className="text-dark text-2rem">Soortgelijke
                        patiënten</CardTitle>
                    <CardSubtitle className="text-muted text-1rem">
                        Dit zijn patiënten uit de trainingsdataset die het
                        meest lijken op uw patiënt
                    </CardSubtitle>
                </CardHeader>
                <CardBody>
                    <Row>
                        <Col>
                            <CardSubtitle
                                className="sim-patients-table-header">Percentage
                                SAD</CardSubtitle>
                            <Similar_Patients_Bar
                                method={"hoeveelheidPatientenMetSAD"}
                                prediction={prediction}
                                aantalPatientenMetSAD={count}
                                aantalPatienten={predictions.length}/>
                            <label>Aantal similar patients met SAD: {count} van
                                de {predictions.length} similar patients heeft
                                SAD</label>
                        </Col>
                    </Row>
                    <Row>
                        <Col>
                            <TableContainer>
                                <Table aria-label="collapsible table">
                                    <TableHead
                                        className="sim-patients-table-head">
                                        <TableRow>
                                            <TableCell/>
                                            <TableCell
                                                className="sim-patients-table-header">Heeft
                                                SAD?</TableCell>
                                            <TableCell
                                                className="text-muted font-weight-bold">Certaintyscore</TableCell>
                                            <TableCell/>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody
                                        className="sim-patients-table-body">
                                        {currentPatients.map((data, index) => (
                                            // eslint-disable-next-line react/jsx-pascal-case
                                            <Expanded_Row key={index}
                                                          details={data}
                                                          prediction={predictions[index].toFixed(2)}
                                                          confidence={confidence[index].toFixed(2)}/>
                                        ))}
                                    </TableBody>
                                </Table>
                                <div style={{
                                    overflow: 'auto',
                                    maxHeight: '200px'
                                }}>
                                    <Pagination>
                                        <PaginationItem
                                            disabled={currentPage <= 0}>
                                            <PaginationLink previous
                                                            onClick={() => setCurrentPage(currentPage - 1)}/>
                                        </PaginationItem>
                                        {[...Array(pageCount)].map((page, i) =>
                                            <PaginationItem
                                                active={i === currentPage}
                                                key={i}>
                                                <PaginationLink
                                                    onClick={() => setCurrentPage(i)}>
                                                    {i + 1}
                                                </PaginationLink>
                                            </PaginationItem>
                                        )}
                                        <PaginationItem
                                            disabled={currentPage >= pageCount - 1}>
                                            <PaginationLink next
                                                            onClick={() => setCurrentPage(currentPage + 1)}/>
                                        </PaginationItem>
                                    </Pagination>
                                </div>
                            </TableContainer>
                        </Col>
                    </Row>
                </CardBody>
            </Card>
        </>
    )
}

export default Similar_Patients_Table