import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import { applyMakeup } from '../../services/api';
import '../Steps.css';

const MakeupStep = () => {
  const { user, skinTone, setMakeup, nextStep, previousStep } = useApp();
  const [showStudio, setShowStudio] = useState(false);
  const [makeupResult, setMakeupResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [options, setOptions] = useState({
    applyLipstick: true,
    applyBlush: true,
    lipstickIntensity: 0.9,
    blushIntensity: 1.0
  });

  const handleSkip = () => {
    console.log('Skipped makeup');
    nextStep();
  };

  const handleTryMakeup = () => {
    if (user.gender === 'male') {
      if (!window.confirm('Makeup is typically for female users. Continue anyway?')) {
        handleSkip();
        return;
      }
    }
    setShowStudio(true);
  };

  const handleApplyMakeup = async () => {
    setLoading(true);
    try {
      const blob = await applyMakeup(skinTone.image, options);
      const url = URL.createObjectURL(blob);
      
      setMakeup({ applied: true, image: url });
      setMakeupResult(url);
      console.log('Makeup applied');
      
      setTimeout(() => nextStep(), 2000);
    } catch (error) {
      console.error('Makeup application failed:', error);
      alert('Failed to apply makeup. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="step-content">
      <h1>MAKEUP TRY-ON</h1>
      <p className="subtitle">Would you like to try makeup virtually?</p>
      
      <div className="card">
        {!showStudio ? (
          <div className="choice-buttons">
            <button className="btn-choice" onClick={handleSkip}>
              <span className="choice-text">Skip Makeup</span>
            </button>
            <button className="btn-choice" onClick={handleTryMakeup}>
              <span className="choice-text">Try Makeup</span>
            </button>
          </div>
        ) : (
          <>
            <div className="makeup-controls">
              <h3>Makeup Options</h3>
              <div className="control-group">
                <label>
                  <input 
                    type="checkbox" 
                    checked={options.applyLipstick}
                    onChange={(e) => setOptions(prev => ({ ...prev, applyLipstick: e.target.checked }))}
                  />
                  Apply Lipstick
                </label>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.1" 
                  value={options.lipstickIntensity}
                  onChange={(e) => setOptions(prev => ({ ...prev, lipstickIntensity: parseFloat(e.target.value) }))}
                />
              </div>
              <div className="control-group">
                <label>
                  <input 
                    type="checkbox" 
                    checked={options.applyBlush}
                    onChange={(e) => setOptions(prev => ({ ...prev, applyBlush: e.target.checked }))}
                  />
                  Apply Blush
                </label>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.1" 
                  value={options.blushIntensity}
                  onChange={(e) => setOptions(prev => ({ ...prev, blushIntensity: parseFloat(e.target.value) }))}
                />
              </div>
              <button 
                className="btn-primary" 
                onClick={handleApplyMakeup}
                disabled={loading}
                style={{ width: '100%', marginTop: 'var(--spacing-md)' }}
              >
                {loading ? 'APPLYING...' : 'APPLY MAKEUP'}
              </button>
            </div>
            
            {makeupResult && (
              <div className="preview-container">
                <img src={makeupResult} alt="Makeup Result" />
              </div>
            )}
          </>
        )}
      </div>

      <div className="action-buttons">
        <button className="btn-secondary" onClick={previousStep}>
          ← BACK
        </button>
      </div>
    </div>
  );
};

export default MakeupStep;

