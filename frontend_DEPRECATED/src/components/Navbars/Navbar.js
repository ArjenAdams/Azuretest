import React, {useEffect, useRef, useState} from "react";
import {Link, useLocation} from "react-router-dom";
import {
    Collapse,
    Container,
    Dropdown,
    DropdownItem,
    DropdownMenu,
    DropdownToggle,
    Nav,
    Navbar,
    NavbarBrand,
    NavbarToggler,
    NavItem
} from "reactstrap";

import routes from "routes.js";

function Header(props) {
    const [isOpen, setIsOpen] = useState(false);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [color, setColor] = useState("transparent");
    const sidebarToggle = useRef();
    const location = useLocation();

    const toggle = () => {
        if (isOpen) {
            setColor("transparent");
        } else {
            setColor("dark");
        }
        setIsOpen(!isOpen);
    };

    const dropdownToggle = (e) => {
        setDropdownOpen(!dropdownOpen);
    };

    const getBrand = () => {
        let brandName = "INNO_AI_Lectoraat-AI";
        routes.map((prop, key) => {
            if (window.location.href.indexOf(prop.layout + prop.path) !== -1) {
                brandName = prop.name;
            }
            return null;
        });
        return brandName;
    };

    const openSidebar = () => {
        document.documentElement.classList.toggle("nav-open");
        sidebarToggle.current.classList.toggle("toggled");
    };

    useEffect(() => {
        if (
            window.innerWidth < 993 &&
            document.documentElement.className.indexOf("nav-open") !== -1
        ) {
            document.documentElement.classList.toggle("nav-open");
            sidebarToggle.current.classList.toggle("toggled");
        }
    }, [location]);

    return (
        // add or remove classes depending if we are on full-screen-maps page or not
        <Navbar
            color={
                props.location.pathname.indexOf("full-screen-maps") !== -1
                    ? "dark"
                    : color
            }
            expand="lg"
            className={
                props.location.pathname.indexOf("full-screen-maps") !== -1
                    ? "navbar-absolute fixed-top"
                    : "navbar-absolute fixed-top " +
                    (color === "transparent" ? "navbar-transparent " : "")
            }
        >
            <Container fluid>
                <div className="navbar-wrapper">
                    <div className="navbar-toggle">
                        <button
                            aria-labelledby={"sidebarToggle"}
                            type="button"
                            ref={sidebarToggle}
                            className="navbar-toggler"
                            onClick={() => openSidebar()}
                        >
                            <span className="navbar-toggler-bar bar1"/>
                            <span className="navbar-toggler-bar bar2"/>
                            <span className="navbar-toggler-bar bar3"/>
                        </button>
                    </div>
                    <NavbarBrand href="/">{getBrand()}</NavbarBrand>
                </div>
                <NavbarToggler onClick={toggle}>
                    <span className="navbar-toggler-bar navbar-kebab"/>
                </NavbarToggler>
                <Collapse isOpen={isOpen} navbar
                          className="justify-content-end">
                    <Nav navbar>
                        <NavItem>
                            <Link to="/" className="nav-link btn-magnify">
                                <i className="nc-icon nc-layout-11"/>
                                <p>
                                    <span
                                        className="d-lg-none d-md-block">Dashboard</span>
                                </p>
                            </Link>
                        </NavItem>
                        <Dropdown
                            nav
                            isOpen={dropdownOpen}
                            toggle={(e) => dropdownToggle(e)}
                        >
                            <DropdownToggle caret nav>
                                <i className="nc-icon nc-settings-gear-65"/>
                                <p>
                                    <span
                                        className="d-lg-none d-md-block">Actions</span>
                                </p>
                            </DropdownToggle>
                            <DropdownMenu end>
                                <DropdownItem onClick={() => {
                                    localStorage.clear();
                                }}>
                                    <Link to="/login" className="nav-link">
                                        <span style={{color: 'black'}}>
                                          logout
                                      </span>
                                    </Link>
                                </DropdownItem>
                            </DropdownMenu>
                        </Dropdown>
                    </Nav>
                </Collapse>
            </Container>
        </Navbar>
    );
}

export default Header;
