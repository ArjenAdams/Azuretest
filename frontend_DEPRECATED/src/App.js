import {HashRouter, Redirect, Route, Switch,} from "react-router-dom";
import React from 'react';
import AuthLayout from "./layouts/Login";
import AdminLayout from "./layouts/Admin";
import "bootstrap/dist/css/bootstrap.css";
import "assets/scss/lectoraat.scss?v=1.3.0";
import "perfect-scrollbar/css/perfect-scrollbar.css";
import "assets/css/componentSheet.css";


function App() {

    return (
        <>
            <HashRouter>
                <Switch>
                    <Route path={"/login"} component={AuthLayout}/>
                    <Route path="/admin" component={(props) => <AdminLayout {...props} />}/>
                    <Redirect to="/admin/patients"/>
                </Switch>
            </HashRouter>

        </>
    );
}

export default App;
