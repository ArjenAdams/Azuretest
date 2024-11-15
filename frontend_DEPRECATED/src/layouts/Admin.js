import React, {useRef, useState} from "react";
// javascript plugin used to create scrollbars on windows
import PerfectScrollbar from "perfect-scrollbar";
import {Route, Switch, useLocation} from "react-router-dom";

import Navbar from "components/Navbars/Navbar.js";
import Sidebar from "components/Sidebar/Sidebar.js";

import routes from "routes.js";

let perfectScroll;

function Dashboard(props) {
    const [backgroundColor] = useState("black");
    const [activeColor] = useState("info");
    const mainPanel = useRef();
    const location = useLocation();

    React.useEffect(() => {
        if (navigator.platform.indexOf("Win") > -1) {
            perfectScroll = new PerfectScrollbar(mainPanel.current);
            document.body.classList.toggle("perfect-scrollbar-on");
        }
        return function cleanup() {
            if (navigator.platform.indexOf("Win") > -1) {
                perfectScroll.destroy();
                document.body.classList.toggle("perfect-scrollbar-on");
            }
        };
    });
    React.useEffect(() => {
        mainPanel.current.scrollTop = 0;
        document.scrollingElement.scrollTop = 0;
    }, [location]);
    return (
        <div className="wrapper">
            <Sidebar
                {...props}
                routes={routes}
                bgColor={backgroundColor}
                activeColor={activeColor}
            />
            <div className="main-panel" ref={mainPanel}>
                <Navbar {...props} />
                <Switch>
                    {routes.map((prop, key) => {
                        return <Route
                            path={prop.layout + prop.path}
                            component={prop.component}
                            key={key}
                        />;
                    })}
                </Switch>
            </div>
        </div>
    );
}

export default Dashboard;
