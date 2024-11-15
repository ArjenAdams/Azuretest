import React from 'react';
import {
    Button,
    Card,
    Form,
    Input,
    Label,
    Modal,
    ModalBody,
    ModalFooter,
    ModalHeader
} from 'reactstrap';
import Checkbox from "@material-ui/core/Checkbox";

const AdvancedOptions = ({
                             modal,
                             toggleModal,
                             handleSearchValueChange,
                             handleSearch
                         }) => {

    const fields = [
        {label: 'Temperature', name: 'temperature', step: 0.1},
        {label: 'Heart Rate', name: 'heart_rate'},
        {label: 'Respiratory Rate', name: 'resp_rate'},
        {label: 'SpO2', name: 'spo2'},
        {label: 'Systolic Blood Pressure (SBP)', name: 'sbp'},
        {label: 'Diastolic Blood Pressure (DBP)', name: 'dbp'},
        {label: 'Mean Blood Pressure (MBP)', name: 'mbp'},
        {label: 'White Blood Cells (WBC)', name: 'wbc', step: 0.1},
        {label: 'Hemoglobin', name: 'hemoglobin', step: 0.1},
        {label: 'Platelet', name: 'platelet'},
        {label: 'Blood Urea Nitrogen (BUN)', name: 'bun'},
        {label: 'Creatinine (Cr)', name: 'cr', step: 0.1},
        {label: 'Glucose (Glu)', name: 'glu'},
        {label: 'Sodium (Na)', name: 'Na'},
        {label: 'Chloride (Cl)', name: 'Cl'},
        {label: 'Potassium (K)', name: 'K', step: 0.1},
        {label: 'Magnesium (Mg)', name: 'Mg', step: 0.1},
        {label: 'Calcium (Ca)', name: 'Ca', step: 0.1},
        {label: 'Phosphate (P)', name: 'P', step: 0.1},
        {label: 'INR', name: 'inr', step: 0.1},
        {label: 'Glasgow Coma Scale (GCS)', name: 'gcs'},
        {label: 'Prothrombin Time (PT)', name: 'pt', step: 0.1},
        {label: 'Partial Thromboplastin Time (PTT)', name: 'ptt', step: 0.1},
        {label: 'Bicarbonate', name: 'bicarbonate'},
        {label: 'Anion Gap', name: 'aniongap'},
        {label: 'SOFA Score', name: 'sofa_score'},
    ];

    const booleanFields = [
        {label: 'Ventilation', name: 'vent'},
        {label: 'CRRT', name: 'crrt'},
        {label: 'Vasopressors', name: 'vaso'},
        {label: 'Sedation', name: 'seda'},
        {label: 'Acute Myocardial Infarction', name: 'ami'},
        {label: 'Chronic Kidney Disease', name: 'ckd'},
        {label: 'Chronic Obstructive Pulmonary Disease', name: 'copd'},
        {label: 'Hypertension', name: 'hyperte'},
        {label: 'Diabetes Mellitus', name: 'dm'},
        {label: 'Acute Kidney Injury', name: 'aki'},
        {label: 'Stroke', name: 'stroke'},
    ];


    const handleSubmit = (event) => {
        handleSearch(event);
        toggleModal();

    }

    const handleChange = (event) => {
        event.preventDefault();

        console.log(event)

        handleSearchValueChange(event);
    }


    return (
        <Card>
            {/* ... other code ... */}
            <Modal isOpen={modal} toggle={toggleModal}>
                <ModalHeader toggle={toggleModal}>Geavanceerde
                    opties</ModalHeader>
                <ModalBody>
                    <Form onSubmit={handleSubmit} id={"AdvancedSearchForm"}>
                        {fields.map((field) => (
                            <div key={field.name}>
                                <Label
                                    htmlFor={field.name}>{field.label}</Label>
                                <Input
                                    type="number"
                                    id={field.name}
                                    name={field.name}
                                    step={field.step || 1}
                                    onChange={handleChange}
                                />
                            </div>
                        ))}
                        {booleanFields.map((field) => (
                            <div key={field.name}>
                                <Label
                                    htmlFor={field.name}>{field.label}</Label>
                                <Checkbox
                                    type={"checkbox"}
                                    id={field.name}
                                    name={field.name}
                                    onChange={handleChange}
                                />
                            </div>
                        ))}
                        <Button color="primary" type="submit"
                                form={"AdvancedSearchForm"}>Search</Button>
                    </Form>
                </ModalBody>
                <ModalFooter>
                    <Button color="secondary"
                            onClick={toggleModal}>Close</Button>
                </ModalFooter>
            </Modal>
            {/* ... other code ... */}
        </Card>
    )
};

export default AdvancedOptions;