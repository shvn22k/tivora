import React, { useState, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import { analyzeSkinTone, detectUndertone } from '../../services/api';
import '../Steps.css';

const SkinToneStep = () => {
  const { setSkinTone, skinTone, nextStep, previousStep } = useApp();
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzed, setAnalyzed] = useState(false);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSkinTone(prev => ({ ...prev, image: file }));
    setPreviewUrl(URL.createObjectURL(file));
    setAnalyzed(false);
  };

  const handleAnalyze = async () => {
    if (!skinTone.image) return;

    setLoading(true);
    try {
      const data = await analyzeSkinTone(skinTone.image);
      
      if (data.success) {
        const undertone = detectUndertone(data.skin_tone_hex);
        setSkinTone(prev => ({
          ...prev,
          hex: data.skin_tone_hex,
          undertone
        }));
        setAnalyzed(true);
        console.log(`Skin tone: ${data.skin_tone_hex}, ${undertone}`);
      }
    } catch (error) {
      console.error('Skin tone analysis failed:', error);
      alert('Failed to analyze skin tone. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="step-content">
      <h1>SKIN TONE ANALYSIS</h1>
      <p className="subtitle">Upload a clear face photo for accurate color matching</p>
      
      <div className="card">
        {!previewUrl ? (
          <div 
            className="upload-zone" 
            onClick={() => fileInputRef.current?.click()}
          >
            <p>Click to upload or drag and drop</p>
            <span className="upload-hint">PNG, JPG up to 10MB</span>
            <input 
              ref={fileInputRef}
              type="file" 
              accept="image/*" 
              hidden 
              onChange={handleFileSelect}
            />
          </div>
        ) : (
          <div className="preview-container">
            <img src={previewUrl} alt="Preview" />
          </div>
        )}

        {previewUrl && !analyzed && (
          <button 
            className="btn-primary" 
            onClick={handleAnalyze}
            disabled={loading}
            style={{ marginTop: 'var(--spacing-md)', width: '100%' }}
          >
            {loading ? 'ANALYZING...' : 'ANALYZE SKIN TONE'}
          </button>
        )}

        {analyzed && (
          <div className="result-box">
            <h3>Analysis Complete</h3>
            <div className="tone-display">
              <div 
                className="tone-color" 
                style={{ background: skinTone.hex }}
              />
              <div className="tone-info">
                <div className="tone-label">Your Skin Tone</div>
                <div className="tone-hex">{skinTone.hex}</div>
                <div className="tone-undertone">{skinTone.undertone} Undertone</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="action-buttons">
        <button className="btn-secondary" onClick={previousStep}>
          ← BACK
        </button>
        <button 
          className="btn-primary" 
          onClick={nextStep}
          disabled={!analyzed}
        >
          CONTINUE →
        </button>
      </div>
    </div>
  );
};

export default SkinToneStep;

