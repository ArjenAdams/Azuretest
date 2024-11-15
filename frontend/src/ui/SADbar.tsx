import { useEffect, useState } from "react";

type SADBarProps = {
    positivePercentage: number;
    hasPreviousPercentage: boolean;
    previousPercentage: number;
};

export const Sadbar = (props: SADBarProps) => {
    const [animatedPercentage, setAnimatedPercentage] = useState(0);
    const [previousPointerRotation, setPreviousPointerRotation] = useState(0);

    useEffect(() => {
        let start: number | null = null;
        const duration = 500;
        const initialPercentage = animatedPercentage;
        const targetPercentage = props.positivePercentage;

        const animate = (timestamp: number) => {
            if (!start) start = timestamp;
            const progress = timestamp - start;

            const newPercentage = Math.min(
                initialPercentage + (progress / duration) * (targetPercentage - initialPercentage),
                targetPercentage
            );

            setAnimatedPercentage(newPercentage);

            if (progress < duration) {
                window.requestAnimationFrame(animate);
            }
        };

        window.requestAnimationFrame(animate);

        if (props.hasPreviousPercentage) {
            setPreviousPointerRotation((props.previousPercentage / 100) * 180);
        }
    }, [props.positivePercentage, props.previousPercentage, props.hasPreviousPercentage]);

    const pointerRotation = (animatedPercentage / 100) * 180;
    const negativePercentage = 100 - animatedPercentage;

    return (
        <div className="w-full px-4">
            <div className="bg-white rounded-[80px] p-1 shadow-md mx-auto flex items-center justify-between">
                <h2 className="ml-16 text-4xl text-right">Heeft SAD?</h2>
                <div className="mx-auto flex items-center">
                    <div className="flex justify-center">
                        <div className="relative w-32 h-16">
                            <svg className="absolute inset-0" viewBox="0 0 100 55">
                                <defs>
                                    <linearGradient id="grad1" x1="0%" y1="100%" x2="100%" y2="100%">
                                        <stop offset="0%" style={{ stopColor: '#a0e0d8', stopOpacity: 1 }} />
                                        <stop offset="100%" style={{ stopColor: '#223c72', stopOpacity: 1 }} />
                                    </linearGradient>
                                </defs>
                                <path
                                    d="M 10 50 A 40 40 0 0 1 90 50"
                                    stroke="url(#grad1)"
                                    strokeWidth="8"
                                    fill="none"
                                    strokeLinecap="round"
                                />
                                {props.hasPreviousPercentage && (
                                    <ellipse
                                        cx="50"
                                        cy="50"
                                        rx="2"
                                        ry="20"
                                        transform={`rotate(${previousPointerRotation - 90} 50 50)  translate(0, -20)`}
                                        fill="black"
                                        fillOpacity="0.3"
                                    />
                                )}
                                <ellipse
                                    cx="50"
                                    cy="50"
                                    rx="2"
                                    ry="20"
                                    transform={`rotate(${pointerRotation - 90} 50 50) translate(0, -20)`}
                                    fill="black"
                                />
                            </svg>
                        </div>
                    </div>

                    <div className="ml-32 text-right">
                        <div className="flex justify-between w-120 text-sm">
                            <div className="text-blue-900 text-2xl">
                                Positief
                            </div>
                            <div className="text-blue-900 text-2xl">
                                Negatief
                            </div>
                        </div>
                        <div className="flex justify-between w-60 text-lg mb-1">
                            <div className="text-blue-900 font-bold text-3xl">
                                {animatedPercentage.toFixed(1)} %
                            </div>
                            <div className="text-blue-900 font-bold text-3xl">
                                {negativePercentage.toFixed(1)} %
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Sadbar;
