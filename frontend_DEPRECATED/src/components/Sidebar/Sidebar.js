import React, {useEffect, useRef, useState} from "react";
import {NavLink} from "react-router-dom";
import {Button, Nav} from "reactstrap";
import {fetchPatientData} from "../../handler/sessionStorageHandler"; // Correctly import fetchPatientData

const logo = require("assets/img/JADS_XAI_LOGO.png");

function Sidebar(props) {
    const [open, setOpen] = useState(false);
    const contentRef = useRef();
    const sidebar = useRef();
    const [patient, setPatient] = useState({});

    const activeRoute = (routeName) => {
        return props.location.pathname.indexOf(routeName) > -1 ? "active" : "";
    };

    const toggle = () => {
        setOpen(!open);
    }

    useEffect(() => {
        const patientData = fetchPatientData();
        if (patientData) {
            setPatient(patientData);
        }

        const handleStorageChange = (event) => {
            if (event.key === 'Patient') {
                const updatedPatientData = JSON.parse(event.newValue);
                setPatient(updatedPatientData);
            }
        };

        const handleCustomEvent = () => {
            const updatedPatientData = fetchPatientData();
            if (updatedPatientData) {
                setPatient(updatedPatientData);
            }
        };

        window.addEventListener('storage', handleStorageChange);
        window.addEventListener('patientDataChange', handleCustomEvent);
        return () => {
            window.removeEventListener('storage', handleStorageChange);
            window.removeEventListener('patientDataChange', handleCustomEvent);
        };
    }, []);


    return (
        <div
            className="sidebar"
            data-color={props.bgColor}
            data-active-color={props.activeColor}
        >
            <div className="logo">
                <a href="/" className="simple-text">
                    <div className="logo-img">
                        <img src={logo} alt="JADS XAI logo" style={{
                            width: '80%',
                            height: '80%',
                            objectFit: 'cover'
                        }}/>
                    </div>
                    <span className="simple-text">JADS XAI</span>
                </a>
            </div>
            <div className="sidebar-wrapper" ref={sidebar}>
                <Nav>
                    {props.routes.map((prop, key) => {
                        if (prop.hideInSidebar) {
                            return null;
                        }
                        if (prop.path === '/teaser/' && fetchPatientData() !== null) {
                            return (
                                <li
                                    className={
                                        activeRoute(prop.path) + (prop.pro ? " active-pro" : "")
                                    }
                                    key={key}
                                >
                                    <NavLink
                                        to={prop.layout + prop.path}
                                        className="nav-link"
                                        activeClassName="active"
                                    >
                                        <i className={prop.icon}/>
                                        <p>{prop.name}</p>
                                    </NavLink>
                                    <Button onClick={toggle}
                                            color={"secondary"}
                                            className="w-100">Features
                                        Patient: {patient.ID}</Button>
                                    <div className="content-parent"
                                         ref={contentRef}
                                         style={open ? {height: contentRef.current.scrollHeight + "px"} : {height: "0px"}}>
                                        <ul className="content">
                                            <li>
                                                <NavLink to='/admin/dashboard'
                                                         className="nav-link"
                                                         activeClassName="active">
                                                    <p>Input Features</p>
                                                </NavLink>
                                            </li>
                                            <li>
                                                <NavLink
                                                    to='/admin/changevalues'>
                                                    <p>Change Values</p>
                                                </NavLink>
                                            </li>
                                            <li>
                                                <NavLink
                                                    to='/admin/counterfactual'>
                                                    <p>Counterfactuals</p>
                                                </NavLink>
                                            </li>
                                            <li>
                                                <NavLink
                                                    to='/admin/similar_patients'>
                                                    <p>Similar Patients</p>
                                                </NavLink>
                                            </li>
                                        </ul>
                                    </div>
                                </li>
                            )
                        }
                        return (
                            <li
                                className={
                                    activeRoute(prop.path) + (prop.pro ? " active-pro" : "")
                                }
                                key={key}
                            >
                                <NavLink
                                    to={prop.layout + prop.path}
                                    className="nav-link"
                                    activeClassName="active"
                                >
                                    <i className={prop.icon}/>
                                    <p>{prop.name}</p>
                                </NavLink>
                            </li>
                        );
                    })}
                </Nav>
            </div>
        </div>
    );
}

export default Sidebar;
