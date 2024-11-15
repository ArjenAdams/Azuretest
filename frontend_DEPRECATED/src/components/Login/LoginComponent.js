import React from 'react';
import {
    Button,
    Card,
    CardBody,
    CardFooter,
    CardSubtitle,
    CardTitle,
    Form,
    FormGroup,
    Input,
    InputGroup,
    InputGroupText,
    Label
} from 'reactstrap';
import {FaKey, FaUserAlt} from 'react-icons/fa';
import logo from '../../assets/img/Erasmus.png';
import './login.css';

function LoginComponent() {
    const handleSubmit = async (event) => {
        event.preventDefault();
        window.location.href = '/';
    };

    return (
        <Card style={{width: '22rem', height: '35rem'}}>
            <CardBody>
                <CardTitle className="text-center text-muted p-5">
                    <div className="logo">
                        <img src={logo} alt="Erasmus MC logo"/>
                    </div>
                    <br/>
                    Dashboard login
                </CardTitle>
                <CardSubtitle className="text-muted text-center">
                    Vul je email en wachtwoord in
                </CardSubtitle>
                <Form className="mb-4" onSubmit={handleSubmit}>
                    <FormGroup className="mb-3">
                        <Label className="text-muted">Email</Label>
                        <InputGroup>
                            <InputGroupText addonType="prepend">
                                <FaUserAlt icon="personIcon"/>
                            </InputGroupText>
                            <Input
                                className="text-muted"
                                type="email"
                                placeholder="Email Address"
                                name="username"
                            />
                        </InputGroup>
                    </FormGroup>

                    <FormGroup className="mb-3">
                        <Label className="text-muted">Wachtwoord</Label>
                        <InputGroup>
                            <InputGroupText addonType="prepend">
                                <FaKey icon="personIcon"/>
                            </InputGroupText>
                            <Input
                                className="text-muted"
                                type="password"
                                placeholder="Password"
                                name="password"
                            />
                        </InputGroup>
                    </FormGroup>
                    <Button type="submit" className="w-100 buttonColor">
                        Login
                    </Button>
                    <CardSubtitle
                        className="text-danger text-center p-3 text-2rem">
                        {}
                    </CardSubtitle>
                </Form>
            </CardBody>
            <CardFooter className="text-muted text-1rem">
                info@example.com
            </CardFooter>
        </Card>
    );
}

export default LoginComponent;
