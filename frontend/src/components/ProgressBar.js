import React from 'react';
import { useApp } from '../context/AppContext';

const steps = [
  { number: 1, label: 'Login' },
  { number: 2, label: 'Skin Tone' },
  { number: 3, label: 'Makeup (Optional)' },
  { number: 4, label: 'Upload Photo' },
  { number: 5, label: 'Recommendations' },
  { number: 6, label: 'Try-On Results' }
];

const ProgressBar = () => {
  const { currentStep } = useApp();

  return (
    <div className="progress-bar">
      {steps.map((step, index) => (
        <React.Fragment key={step.number}>
          <div 
            className={`progress-step ${
              step.number === currentStep ? 'active' : 
              step.number < currentStep ? 'completed' : ''
            }`}
          >
            <div className="step-number">{step.number}</div>
            <span>{step.label}</span>
          </div>
          {index < steps.length - 1 && <div className="progress-line" />}
        </React.Fragment>
      ))}
    </div>
  );
};

export default ProgressBar;

