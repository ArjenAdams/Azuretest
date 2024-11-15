import Dashboard from "./views/Dashboard.js";
import Changevalues from "./views/Changevalues.js";
import Patient from "./views/Patient.js";
import Overview from "./views/Overview";
import Counterfactual from "./views/Counterfactual";
import PatientInformation from "./views/PatientInformation";
import Similar_Patients from "./views/Similar_Patients";
import Teaser from "./views/Teaser";
import Shap_Container from "./views/Shap_Container";

const routes = [
    {
        path: "/overview",
        name: "Overview",
        icon: "nc-icon nc-chart-pie-36",
        component: Overview,
        layout: "/admin",
        hideInSidebar: true
    },
    {
        path: "/dashboard",
        name: "Dashboard",
        icon: "nc-icon nc-bank",
        component: Dashboard,
        layout: "/admin",
        hideInSidebar: true
    },
    {
        path: "/patients",
        name: "Patients",
        icon: "nc-icon nc-single-02",
        component: Patient,
        layout: "/admin"
    },
    {
        path: "/changevalues",
        name: "Change values",
        icon: "nc-icon nc-single-02",
        component: Changevalues,
        layout: "/admin",
        hideInSidebar: true
    },
    {
        path: "/counterfactual",
        name: "Counterfactual",
        icon: "nc-icon nc-single-02",
        component: Counterfactual,
        layout: "/admin",
        hideInSidebar: true
    },

    {
        path: "/similar_patients",
        name: "Similar Patients",
        icon: "nc-icon nc-single-02",
        component: Similar_Patients,
        layout: "/admin",
        hideInSidebar: true
    },
    {
        path: "/patient/:id",
        name: ":id",
        icon: "nc-icon nc-single-02",
        component: PatientInformation,
        layout: "/admin",
        hideInSidebar: true,
    },
    {
        path: "/teaser/",
        name: "Dashboard",
        icon: "nc-icon nc-bank",
        component: Teaser,
        layout: "/admin",
    },
    {
        path: "/shap",
        name: "Shap Grafieken",
        icon: "nc-icon nc-bank",
        component: Shap_Container,
        layout: "/admin"
    }

];

export default routes;
