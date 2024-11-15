import {useCallback, useEffect, useState} from 'react';
import axios from 'axios';

let cache = {};

export const useFetch = (url, method = 'get', enableCache = false, cacheTime = 5, postData) => {
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [reload, setReload] = useState(false);

    const fetchData = useCallback(async () => {
        setIsLoading(true);
        try {
            if (cache[url] && !reload && enableCache) {
                const now = Date.now();
                const timeElapsed = (now - cache[url].timestamp) / (1000 * 60); // in minutes

                if (timeElapsed < cacheTime) {
                    console.log('Using cache for url:', url); // Add this line to check if cached data is being used
                    setData(cache[url].data);
                    setIsLoading(false);
                    return;
                }
            }

            let response;
            if (method === 'get') {
                response = await axios.get(url);
                setData(response.data);
            } else if (method === 'post') {
                response = await axios.post(url, postData);
                setData(response.data);
            }
            if (enableCache) {
                console.log('Saving to cache for url:', url); // Add this line to check if data is being saved to cache
                cache[url] = {data: response.data, timestamp: Date.now()};
            }

            setReload(false); // reset the reload state after fetching

        } catch (error) {
            setError(error);
        } finally {
            setIsLoading(false);
        }
    }, [url, reload, enableCache, cacheTime]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    return {data, isLoading, error, refetch: () => setReload(true)};
};

