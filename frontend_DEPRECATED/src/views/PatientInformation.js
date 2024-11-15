import React from 'react';
import {
    Button,
    Card,
    CardBody,
    CardHeader,
    CardSubtitle,
    CardTitle,
    Col,
    Row,
    Table
} from 'reactstrap';
import {Link, useLocation} from 'react-router-dom';
import {setPatientData} from "../handler/sessionStorageHandler";
import {isFloat32Array, isFloat64Array} from "util/support/types";

function PatientInformation() {
    const location = useLocation();
    const {patient} = location.state || {};
    setPatientData(patient);

    return (
        <div className="content">
            <Card>
                <CardHeader>
                    <CardTitle>
                        <h1 className="text-dark font-weight-bold text-2rem">Patient
                            Monitor</h1>
                    </CardTitle>
                    <CardSubtitle>
                        <Link to={{pathname: `/admin/patients`}}>
                            <Button color="primary">Back</Button>
                        </Link>
                        <Link
                            to={{pathname: `/admin/teaser`, state: {patient}}}>
                            <Button color="primary">More Info</Button>
                        </Link>
                    </CardSubtitle>
                </CardHeader>
                <CardBody>
                    <Row>
                        <Col md="12">
                            <h3 className="text-dark">Basic Information</h3>
                            <Table hover className="bg-light">
                                <tbody>
                                <tr>
                                    <td>
                                        <strong>Name:</strong>
                                    </td>
                                    <td>{patient?.name || ' '}</td>
                                </tr>
                                <tr>
                                    <td>
                                        <strong>Patient ID:</strong>
                                    </td>
                                    <td>{patient?.patient_id || ' '}</td>
                                </tr>
                                <tr>
                                    <td>
                                        <strong>Birthtime:</strong>
                                    </td>
                                    <td>{patient?.birthtime || ' '}</td>
                                </tr>
                                </tbody>
                            </Table>
                        </Col>
                    </Row>
                    <Row>
                        <Col md="6">
                            <Table hover className="bg-light">
                                <thead>
                                <tr>
                                    <th colSpan="2">
                                        <h6 className="text-dark">Patient
                                            Information</h6>
                                    </th>
                                </tr>
                                </thead>
                                <tbody>
                                {Object.entries(patient || {}).map(([key, value], index) => {
                                    if (index < Object.entries(patient).length / 2) {
                                        return (
                                            <tr key={key}>
                                                {console.log(key)}
                                                <td>
                                                    <strong>{key}:</strong>
                                                </td>
                                                {typeof (value) === "number" && value.toString().includes(".") ? (
                                                    <td>{value.toFixed(2)}</td>
                                                ) : (
                                                    <td>{value}</td>
                                                )}
                                            </tr>
                                        );
                                    }
                                    return null;
                                })}
                                </tbody>
                            </Table>
                        </Col>
                        <Col md="6">
                            <Table hover className="bg-light">
                                <thead>
                                <tr>
                                    <th colSpan="2">
                                        <h6 className="text-dark">Patient
                                            Information</h6>
                                    </th>
                                </tr>
                                </thead>
                                <tbody>
                                {Object.entries(patient || {}).map(([key, value], index) => {
                                    if (index >= Object.entries(patient).length / 2) {
                                        return (
                                            <tr key={key}>
                                                <td>
                                                    <strong>{key}:</strong>
                                                </td>
                                                <td>{value}</td>
                                            </tr>
                                        );
                                    }
                                    return null;
                                })}
                                </tbody>
                            </Table>
                        </Col>
                    </Row>
                </CardBody>
            </Card>
        </div>
    );
}

export default PatientInformation;
