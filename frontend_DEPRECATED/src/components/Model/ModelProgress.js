import React, {useState} from 'react';
import {Progress} from 'reactstrap';

const ModelProgress = ({percentage}) => {
    const [barValue, setBarValue] = useState(percentage);

    const indicatorStyle = {
        height: '100%',
        width: '1px',
        background: 'black',
        position: 'absolute',
        top: '0',
        left: `${(barValue / 100) * 100}%`,
    };

    return (
        <div>
            <div style={{position: 'relative'}}>
                <Progress multi>
                    <Progress bar color="danger" value={75} animated={true}/>
                    <Progress bar color="success" value={25} animated={true}/>
                </Progress>
                <div style={indicatorStyle}></div>
                <div style={{
                    position: 'absolute',
                    top: '100%',
                    left: `${(barValue / 100) * 100}%`,
                    transform: 'translateX(-50%)'
                }}>
                    {`${barValue}%`}
                </div>
            </div>
        </div>
    );
};

export default ModelProgress;