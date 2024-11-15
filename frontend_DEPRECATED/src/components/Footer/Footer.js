import React from "react";
import {Container, Row} from "reactstrap";
import PropTypes from "prop-types";

function Footer(props) {
    return (
        <footer
            className={"footer" + (props.default ? " footer-default" : "")}>
            <Container fluid={props.fluid}>
                <Row>
                    <nav className="footer-nav">
                        <ul>
                            INNO_AI_Lectoraat-AI
                        </ul>
                    </nav>
                    <div className="credits ml-auto">
                    </div>
                </Row>
            </Container>
        </footer>
    );
}

Footer.propTypes = {
    default: PropTypes.bool,
    fluid: PropTypes.bool
};

export default Footer;
