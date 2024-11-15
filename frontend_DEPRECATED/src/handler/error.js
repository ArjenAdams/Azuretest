import React from 'react';
import {Alert, Button} from 'reactstrap';

function ErrorScreen({errorMessage, buttonFunction, isPressed = false}) {
    function reload() {
        buttonFunction();
        isPressed = true;
    }

    return (
        <div className="content" style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh'
        }}>
            <Alert color="danger">
                An error occurred: {errorMessage}
                <br/>
                {buttonFunction && (
                    <>
                        <br/>
                        <Button color="primary"
                                onClick={reload}>Reload</Button>
                    </>
                )}
            </Alert>
        </div>
    );
}

export default ErrorScreen;
