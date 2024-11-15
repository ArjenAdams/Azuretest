import React from "react";

function GenerateShapPlot({shap_image}) {
    return (
        <>
            <div className={"content"}>
                <head>
                    <meta charset="UTF-8"/>
                    <meta name="viewport"
                          content="width=device-width, initial-scale=1.0"/>
                    <title>SHAP Plot</title>
                </head>
                <body>
                {shap_image !== undefined && shap_image !== null && shap_image !== "" ? (
                    <div id="shap-plot-container">
                        <img className={"xgboost_image"}
                             src={`data:image/png;base64,${shap_image}`}
                             alt={"explanation graph"}/>
                    </div>
                ) : (
                    <p>Loading shap data...</p>
                )}
                </body>
            </div>
        </>
    );
}

export default GenerateShapPlot;
