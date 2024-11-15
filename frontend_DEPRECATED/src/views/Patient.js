import React, {useEffect, useState} from 'react';
import SearchPatient from '../components/Patient/SearchPatient.js';
import PatientTable from '../components/Patient/PatientTable.js';
import {useFetch} from "../handler/useFetch";
import API_URL from "../handler/api.js";
import LoadingScreen from "../handler/loading";
import ErrorScreen from "../handler/error";
import {backendConnection} from "../handler/backendConnection";

const Patient = () => {
    const {
        data,
        isLoading,
        error,
        refetch
    } = useFetch(API_URL + "data", 'get', true, 10);

    const [patientsData, setPatientsData] = useState([]);
    const [filteredPatients, setFilteredPatients] = useState([]);
    const [searchValues, setSearchValues] = useState({});
    const [pressed, setPressed] = useState(false);

    useEffect(() => {
        if (data && data.length > 0) {
            const headers = ['ID', 'AGE', 'WEIGHT', 'GENDER', 'ETHNICITY', 'ICU', "SAD"];


            const initialSearchValues = headers.reduce((acc, header) => {
                acc[header] = {value: ''};
                acc[header].value = null;
                return acc;
            }, {});

            setSearchValues(initialSearchValues);
            setPatientsData(data);
            setFilteredPatients(data);  // add this line

        }
    }, [data]);

    const handleSearchValueChange = (event) => {
        let setName = null;
        let setValue = null;

        console.log(event.target.type)

        if (event.target.type === "checkbox") {
            console.log(event.target.checked)
            let tf = event.target.checked;
            setName = event.target.name;
            setValue = tf;
        } else {
            let {name, type, value,} = event.target;
            setName = name;
            setValue = value;
            if (value === "" || value === null) {
                setValue = null
            }
        }
        setSearchValues((prevValues) => ({
            ...prevValues,
            [setName]: setValue
        }));
    };


    const handleSearch = async (e) => {
        console.log(`Searching for patient with criteria: `, searchValues);

        e.preventDefault();

        let searchValueData = null;

        try {
            searchValueData = await backendConnection(searchValues)
        } catch (error) {
            console.error("Error fetching data from database", error)
        }

        setFilteredPatients(searchValueData);
    };

    if (isLoading) return <LoadingScreen/>;
    if (error) return <ErrorScreen errorMessage={error.message}
                                   buttonFunction={refetch}
                                   isPressed={false}/>;

    return (
        <>
            <div className={"content"}>
                <SearchPatient
                    searchValues={searchValues}
                    handleSearchValueChange={handleSearchValueChange}
                    handleSearch={handleSearch}
                />
                <PatientTable patients={filteredPatients}/>
            </div>
        </>
    );
};

export default Patient;