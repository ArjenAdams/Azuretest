import API_URL from "./api";

export const backendConnection = async (searchValues) => {


    searchValues = Object.entries(searchValues).reduce((acc, [key, value]) => {
        if (value.value !== null) {
            acc[key] = value;
        }
        return acc;
    }, {});


    const response = await fetch(API_URL + "searchvalues", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(searchValues)
    });
    return await response.json();

}