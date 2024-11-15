import {Button, Card, Form, Input, Label, Modal, ModalBody, ModalFooter, ModalHeader} from "reactstrap";
import React, {useState} from "react";
import Checkbox from "@material-ui/core/Checkbox";


const AdvancedFilter = ({modal, toggleModal, handleSubmit}) => {
    const [open, setOpen] = useState(false);


     const fields = [
         { label: 'Age', name: 'age' },
    { label: 'Weight', name: 'weight' },
         { label: 'Gender', name: 'gender' },
         { label: 'ICU', name: 'icu' },
         { label: 'Ethnicity', name: 'ethnicity'},

  { label: 'Temperature', name: 'temperature', step: 0.1 },
  { label: 'Heart Rate', name: 'heart_rate' },
  { label: 'Respiratory Rate', name: 'resp_rate' },
  { label: 'SpO2', name: 'spo2' },
  { label: 'Systolic Blood Pressure (SBP)', name: 'sbp' },
  { label: 'Diastolic Blood Pressure (DBP)', name: 'dbp' },
  { label: 'Mean Blood Pressure (MBP)', name: 'mbp' },
  { label: 'White Blood Cells (WBC)', name: 'wbc', step: 0.1 },
  { label: 'Hemoglobin', name: 'hemoglobin', step: 0.1 },
  { label: 'Platelet', name: 'platelet' },
  { label: 'Blood Urea Nitrogen (BUN)', name: 'bun' },
  { label: 'Creatinine (Cr)', name: 'cr', step: 0.1 },
  { label: 'Glucose (Glu)', name: 'glu' },
  { label: 'Sodium (Na)', name: 'Na' },
  { label: 'Chloride (Cl)', name: 'Cl' },
  { label: 'Potassium (K)', name: 'K', step: 0.1 },
  { label: 'Magnesium (Mg)', name: 'Mg', step: 0.1 },
  { label: 'Calcium (Ca)', name: 'Ca', step: 0.1 },
  { label: 'Phosphate (P)', name: 'P', step: 0.1 },
  { label: 'INR', name: 'inr', step: 0.1 },
        { label: 'Glasgow Coma Scale (GCS)', name: 'gcs' },
  { label: 'Prothrombin Time (PT)', name: 'pt', step: 0.1 },
  { label: 'Partial Thromboplastin Time (PTT)', name: 'ptt', step: 0.1 },
  { label: 'Bicarbonate', name: 'bicarbonate' },
  { label: 'Anion Gap', name: 'aniongap' },
  { label: 'SOFA Score', name: 'sofa_score' },
          { label: 'Ventilation', name: 'vent' },
          { label: 'CRRT', name: 'crrt' },
          { label: 'Vasopressors', name: 'vaso' },
          { label: 'Sedation', name: 'seda' },
          { label: 'Acute Myocardial Infarction', name: 'ami' },
          { label: 'Chronic Kidney Disease', name: 'ckd' },
          { label: 'Chronic Obstructive Pulmonary Disease', name: 'copd' },
          { label: 'Hypertension', name: 'hyperte' },
          { label: 'Diabetes Mellitus', name: 'dm' },
          { label: 'Acute Kidney Injury', name: 'aki' },
          { label: 'Stroke', name: 'stroke' },
    ];

    let filterList = []

     const handleFilterSubmit = (event) => {
        event.preventDefault();
        handleSubmit(event, filterList);
        toggleModal();

    };

     const handleChange = (event) => {
         event.preventDefault();
         if (filterList.includes(event.target.name)){
             console.log(event.target.name)
             filterList = filterList.filter(item => item !== event.target.name)
         }else {
             filterList.push(event.target.name);
         }
         console.log(filterList)

     }
    return (
        <Card>
            <Modal isOpen={modal} toggle={toggleModal}>
                <ModalHeader toggle={toggleModal}>Geavanceerde filter</ModalHeader>
                <ModalBody>
                    <Form onSubmit={handleFilterSubmit} id={"AdvancedFilterForm"}>
                        {fields.map((field) => (
                            <div key={field.name}>
                                <Label htmlFor={field.name}>{field.label}</Label>
                                <Checkbox type={"checkbox"} id={field.name} name={field.name} onChange={handleChange}/>
                            </div>
                        ))}
                    </Form>
                    <Button color="primary" type="submit" form={"AdvancedFilterForm"}>Pas toe</Button>
                </ModalBody>
                <ModalFooter>
                    <Button color="secondary"  onClick={toggleModal}>Close</Button>
                </ModalFooter>
            </Modal>
        </Card>
    );

}

export default AdvancedFilter;

