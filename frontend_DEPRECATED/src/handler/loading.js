import React from 'react';
import {Spinner} from 'reactstrap';

const LoadingScreen = () => {
    return (
        <div className="content" style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh',
            flexDirection: 'column'
        }}>
            <Spinner
                style={{width: '3rem', height: '3rem', marginBottom: '1rem'}}
                color="primary" children={false}/>
            <div>Loading...</div>
        </div>
    );
}

export default LoadingScreen;
