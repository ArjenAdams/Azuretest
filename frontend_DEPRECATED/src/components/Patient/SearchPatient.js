import React from 'react';
import {
    Button,
    Card,
    CardBody,
    CardFooter,
    CardHeader,
    CardTitle,
    Col,
    Form,
    FormGroup,
    Input,
    Label,
    Row
} from 'reactstrap';
import Slider from "@material-ui/core/Slider";
import Checkbox from "@material-ui/core/Checkbox";
import AdvancedOptions from "../../views/AdvancedOptions";

function changeStorage() {
    sessionStorage.setItem("InvalidValueAlert", 'false')
}

const numberMarksAge = [
    {
        value: 0,
        label: "0"
    },
    {
        value: 100,
        label: "100"
    }
];
const numberMarksWeight = [
    {
        value: 0,
        label: "0"
    },
    {
        value: 200,
        label: "200"
    }
];


const SearchPatient = ({
                           searchValues,
                           handleSearchValueChange,
                           handleSearch
                       }) => {
    const [ageValue, setAgeValue] = React.useState([25, 75]);
    const [weightValue, setWeightValue] = React.useState([50, 150]);
    const [isSAD, setIsSAD] = React.useState(false);
    const [modal, setModal] = React.useState(false);

    const toggleModal = () => {
        setModal(!modal);
    };

    const handleSearchValueChangeID = (event) => {
        if (event.target.value === "") {
            event.target.value = {value: null}
        }
        handleSearchValueChange(event);

    }

    const handleSearchAdvanced = (e) => {
        handleSearch(e);
    }

    const handleSearchValueChangeAdvanced = (event) => {
        handleSearchValueChange(event);
    }

    const handleAGEChange = (event, newValue) => {
        setAgeValue(newValue);
        event.target.value = newValue;
        event.target.name = "AGE";
        handleSearchValueChange(event)
    };
    const handleWEIGHTChange = (event, newValue) => {
        setWeightValue(newValue);
        event.target.value = newValue;
        event.target.name = "WEIGHT";
        handleSearchValueChange(event)
    };

    const handleSADChange = (event) => {
        event.target.type = "checkbox";
        event.target.name = "SAD";
        if (isSAD) {
            event.target.checked = false;
            setIsSAD(false);
            handleSearchValueChange(event)
        } else {
            event.target.checked = true;
            setIsSAD(true);
            handleSearchValueChange(event)
        }
    }


    return (
        <Card>
            <CardHeader>
                <CardTitle tag="h4">Search Patients</CardTitle>
            </CardHeader>
            <CardBody>
                <Form onSubmit={handleSearch} id={"searchform"}>
                    <Row>
                        <Col md={2} key="ID">
                            <FormGroup check>
                                <Label check>ID</Label>
                                <Input type="number" name="ID"
                                       style={{width: '150px'}}
                                       onChange={handleSearchValueChangeID}/>
                            </FormGroup>
                        </Col>
                        <Col md={2} key="AGE">
                            <FormGroup check>
                                <Label check>AGE</Label>
                                <Slider
                                    color="#51cbce"
                                    valueLabelDisplay="auto"
                                    onChange={handleAGEChange}
                                    marks={numberMarksAge}
                                    value={ageValue}
                                />
                            </FormGroup>
                        </Col>
                        <Col md={3} key="WEIGHT">
                            <FormGroup check>
                                <Label check>WEIGHT</Label>
                                <Slider
                                    color="#51cbce"
                                    valueLabelDisplay="auto"
                                    onChange={handleWEIGHTChange}
                                    marks={numberMarksWeight}
                                    value={weightValue}
                                    min={0}
                                    max={200}
                                />
                            </FormGroup>
                        </Col>

                        <Col md={1} key="GENDER">
                            <FormGroup check>
                                <Label check>GENDER</Label>
                                <select name="GENDER"
                                        onChange={handleSearchValueChange}>
                                    <option selected disabled>Select Gender
                                    </option>
                                    <option value="MALE">MALE</option>
                                    <option value="FEMALE">FEMALE</option>
                                    <option value="OTHER">OTHER</option>
                                </select>
                            </FormGroup>
                        </Col>
                        <Col md={1} key="ETHNICITY">
                            <FormGroup check>
                                <Label check>ETHNICITY</Label>
                                <select name="ETHNICITY"
                                        onChange={handleSearchValueChange}>
                                    <option selected disabled>Select
                                        Ethnicity
                                    </option>
                                    <option value="WHITE">WHITE</option>
                                    <option value="BLACK">BLACK</option>
                                    <option value="HISPANIC">HISPANIC</option>
                                    <option value="AISAN">ASIAN</option>
                                    <option value="OTHER">OTHER</option>
                                    <option value="unknown">UNKNOWN</option>
                                </select>
                            </FormGroup>
                        </Col>
                        <Col md={1} key="ICU">
                            <FormGroup check>
                                <Label check>ICU</Label>
                                <select name="ICU"
                                        onChange={handleSearchValueChange}>
                                    <option selected disabled>Select ICU
                                    </option>
                                    <option value="CCU">CCU</option>
                                    <option value="CVICU">CVICU</option>
                                    <option value="MICU">MICU</option>
                                    <option value="MICU/SICU">MICU/SICU
                                    </option>
                                    <option value="NICU">NICU</option>
                                    <option value="SICU">SICU</option>
                                    <option value="TSICU">TSICU</option>
                                </select>
                            </FormGroup>
                        </Col>
                        <Col md={1} key="SAD">
                            <FormGroup check>
                                <div key={"SAD"}>
                                    <Label check>SAD</Label>
                                    <Checkbox type={"checkbox"} name={"SAD"}
                                              checked={isSAD}
                                              onClick={handleSADChange}/>
                                </div>
                            </FormGroup>
                        </Col>
                    </Row>
                </Form>
                <CardFooter>
                    <Button color="primary" type="button"
                            style={{marginRight: '20px'}} onClick={toggleModal}
                            className="mt-3">Geavanceerde opties</Button>
                    <Button color="primary" form={"searchform"} type="submit"
                            onClick={changeStorage}
                            className="mt-3">Search</Button>
                </CardFooter>
                <AdvancedOptions modal={modal} toggleModal={toggleModal}
                                 handleSearchValueChange={handleSearchValueChangeAdvanced}
                                 handleSearch={handleSearchAdvanced}/>
            </CardBody>
        </Card>
    )
}


export default SearchPatient;
