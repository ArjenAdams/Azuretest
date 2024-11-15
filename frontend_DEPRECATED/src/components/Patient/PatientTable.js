import React, {useState} from 'react';
import {
    Button,
    Card,
    CardBody,
    CardHeader,
    CardTitle,
    Pagination,
    PaginationItem,
    PaginationLink,
    Table
} from "reactstrap";
import {Link} from "react-router-dom";

const PatientTable = ({patients}) => {
    // Alle headers voor als je andere data wilt laten zien.
    // const headers = ['ID','age', 'weight', 'gender', 'temperature', 'heart_rate', 'resp_rate',
    //                     'spo2', 'sbp', 'dbp', 'mbp', 'wbc', 'hemoglobin', 'platelet', 'bun', 'cr', 'glu',
    //                     'Na', 'Cl', 'K', 'Mg', 'Ca', 'P', 'inr', 'pt', 'ptt', 'bicarbonate', 'aniongap', 'gcs', 'vent',
    //                     'crrt', 'vaso', 'seda', 'sofa_score', 'ami', 'ckd', 'copd', 'hyperte', 'dm', 'aki', 'stroke',
    //                     'AISAN', 'BLACK', 'HISPANIC', 'OTHER', 'WHITE', 'unknown', 'CCU', 'CVICU', 'MICU', 'MICU/SICU',
    //                     'NICU', 'SICU', 'TSICU'
    //                     ]
    //'CCU', 'CVICU', 'MICU', 'MICU/SICU',
    //                         'NICU', 'SICU', 'TSICU'

    // Gekozen headers
    const headers = ['ID', 'age', 'weight', 'gender',
        'ETHNICITY', 'ICU', 'SAD'
    ]
    const [currentPage, setCurrentPage] = useState(0);
    const pageSize = 20; // Change this to your desired page size
    const pageCount = Math.ceil(patients.length / pageSize);
    let currentPatients
    if (typeof patients === "string") {
        const arrayOfObjects = JSON.parse(patients);
        currentPatients = arrayOfObjects.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
    } else {
        try {
            currentPatients = patients.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
            sessionStorage.setItem("InvalidValueAlert", 'false')
        } catch (e) {
            if (sessionStorage.getItem("InvalidValueAlert") === 'false') {
                sessionStorage.setItem("InvalidValueAlert", 'true')
                //alert("Niet alle ingevoerde waardes zijn geldig")

            }
            return null
        }
    }
    console.log(currentPatients)


    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-dark font-weight-bold text-2rem"
                           tag={"h4"}>Search Results</CardTitle>
            </CardHeader>
            <CardBody>
                <Table responsive>
                    <thead className="text-primary">
                    <tr>
                        {headers.map((header, index) => (
                            <th key={index}>{header.toLowerCase()}</th>
                        ))}
                        <th>Details</th>
                    </tr>
                    </thead>
                    <tbody>
                    {
                        currentPatients.map((patient, index) => (
                            <tr key={index}>
                                {headers.map((header, i) => (
                                    <td key={i}>{
                                        checkIfGender(header, patient)
                                    }</td>
                                ))}
                                <td>
                                    <Link to={{
                                        pathname: `/admin/patient/${patient.ID}`,
                                        state: {patient}
                                    }}>
                                        <Button color="primary">More
                                            Info</Button>
                                    </Link>
                                </td>
                            </tr>
                        ))
                    }
                    </tbody>
                </Table>
                <div style={{overflow: 'auto', maxHeight: '200px'}}>
                    <Pagination>
                        <PaginationItem disabled={currentPage <= 0}>
                            <PaginationLink previous
                                            onClick={() => setCurrentPage(currentPage - 1)}/>
                        </PaginationItem>
                        {[...Array(pageCount)].map((page, i) =>
                            <PaginationItem active={i === currentPage} key={i}>
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
            </CardBody>
        </Card>
    )
}

function checkIfGender(header, patient) {
    if (header === "weight") {
        return patient[header].toFixed(1) + " kg"
    }
    if (header === "ICU") {
        return handleICU(patient);
    }
    if (header === "ETHNICITY") {
        return handleEthnicity(patient);
    }
    if (patient[header] === true) {
        return "True"
    }
    if (patient[header] === false) {
        return "False"
    }
    if (header === "gender") {
        return genderConverter(patient.gender)
    } else {
        return patient[header]
    }
}


function handleICU(patient) {
    if (patient.CCU) {
        return 'CCU'
    }
    if (patient.CVICU) {
        return 'CVICU'
    }
    if (patient.MICU) {
        return 'MICU'
    }
    if (patient.NICU) {
        return 'NICU'
    }
    if (patient.SICU) {
        return 'SICU'
    }
    if (patient.TSICU) {
        return 'TSICU'
    } else {
        return 'MICU/SICU'
    }
}

function handleEthnicity(patient) {
    if (patient.AISAN) {
        return "Asian"
    }
    if (patient.HISPANIC) {
        return "Hispanic"
    }
    if (patient.WHITE) {
        return "White"
    }
    if (patient.BLACK) {
        return "Black"
    }
    if (patient.OTHER) {
        return "Other"
    }
    if (patient.unknown) {
        return "Unknown"
    }
}

function genderConverter(genderNumber) {
    // Gender converten naar letters
    let gender = 'M'
    if (genderNumber === 0) {
        gender = 'F'
        return "Female"
    }
    return "Male"
}

export default PatientTable;
