export const setPatientData = (patient) => {
    sessionStorage.setItem("Patient", JSON.stringify(patient))
    const event = new Event('patientDataChange');
    window.dispatchEvent(event);
}

export const fetchPatientData = () => {
    const patientString = sessionStorage.getItem("Patient");
    if (patientString === null || patientString === undefined) {
        return null;
    }
    return JSON.parse(patientString);
}

export const setCounterfactualData = (counterfactual) => {
    sessionStorage.setItem("Counterfactual", JSON.stringify(counterfactual))
}
export const setSimilarPatientsData = (similarPatients) => {
    sessionStorage.setItem("SimilarPatients", JSON.stringify(similarPatients))
}
export const fetchSimilarPatientsData = () => {
    const similarPatientsString = sessionStorage.getItem("SimilarPatients");
    if (similarPatientsString === null || similarPatientsString === undefined) {
        return null;
    }
    return JSON.parse(similarPatientsString);

}
export const fetchCounterfactualData = () => {
    const counterfactualString = sessionStorage.getItem("Counterfactual");
    if (counterfactualString === null || counterfactualString === undefined) {
        return null;
    }
    return JSON.parse(counterfactualString);

}

export const setPredictionData = (prediction) => {
    sessionStorage.setItem("Prediction", JSON.stringify(prediction))
}

export const fetchPredictionData = () => {
    const predictionString = sessionStorage.getItem("Prediction");
    if (predictionString === null || predictionString === undefined) {
        return null;
    }
    return JSON.parse(predictionString);
}