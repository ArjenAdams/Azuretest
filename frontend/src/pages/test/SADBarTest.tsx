"use client";

import React, { useState } from 'react';
import Sadbar from '../../ui/SADbar';

const SadBarTest = () => {
    const [positivePercentage, setPositivePercentage] = useState(63.8);

    const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setPositivePercentage(parseFloat(event.target.value));
    };

    return (
        <div className="flex flex-col items-center w-full h-screen bg-cyan-800">
            <Sadbar positivePercentage={positivePercentage} hasPreviousPercentage={false} previousPercentage={46.5} />

            <div className="mt-6">
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={positivePercentage}
                    onChange={handleSliderChange}
                    className="slider w-64"
                />
            </div>

            <Sadbar positivePercentage={positivePercentage} hasPreviousPercentage={true} previousPercentage={46.5} />

        </div>
    );
}

export default SadBarTest;