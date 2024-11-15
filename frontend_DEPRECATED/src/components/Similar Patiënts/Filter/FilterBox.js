import React, {useState} from 'react';
import {Checkbox, FormControlLabel} from "@material-ui/core";
import FilterSlider from "./FilterSliders.js";
import {Dropdown, DropdownMenu, DropdownToggle} from "reactstrap";

function FilterBox(props) {
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [checkedA, setCheckedA] = React.useState(true);
    const [checkedB, setCheckedB] = React.useState(true);

    const dropdownToggle = (e) => {
        setDropdownOpen(!dropdownOpen);
    };
    const handleCheckboxChange = (name) => (event) => {
        if (name === "checkedA") {
            setCheckedA(event.target.checked);
        } else if (name === "checkedB") {
            setCheckedB(event.target.checked);
        }
    }


    return (
        <div>
            <Dropdown isOpen={dropdownOpen} toggle={dropdownToggle}>
                <DropdownToggle caret nav>Show Filters
                </DropdownToggle>
                <DropdownMenu>
                    <div className="checkboxes">
                        <FormControlLabel control={
                            <Checkbox
                                checked={checkedA}
                                onChange={handleCheckboxChange("checkedA")}
                                name="checkedA"
                                color="primary"
                            />
                        } label="Man"
                        />
                    </div>
                    <div className="checkboxes">
                        <FormControlLabel control={
                            <Checkbox
                                checked={checkedB}
                                onChange={handleCheckboxChange("checkedB")}
                                name="checkedB"
                                color="primary"
                            />
                        } label="Vrouw"
                        />
                    </div>
                    <div className="slider">
                        <FilterSlider/>
                    </div>
                </DropdownMenu>
            </Dropdown>
        </div>
    )
}

export default FilterBox;