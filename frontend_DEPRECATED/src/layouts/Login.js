import * as React from 'react';
import {useEffect} from 'react';
import LoginComponent from '../components/Login/LoginComponent';
import '../assets/css/loginStyle.css';


function Login() {
    useEffect(() => {
        document.body.classList.add('bodyLogin');

        return () => {
            document.body.classList.remove('bg-black');
        };
    });
    return (
        <div className="d-flex justify-content-center align-items-center"
             style={{height: '100vh'}}>
            <LoginComponent/>
        </div>
    );

}

export default Login;
