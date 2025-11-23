import React, { useEffect, useState, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import { generateTryOn, mapCategory } from '../../services/api';
import '../Steps.css';

const ResultsStep = () => {
  const {
    bodyPhoto,
    recommendations,
    selectedGarments,
    tryonResults,
    setTryonResults,
    resetApp
  } = useApp();
  
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const hasStarted = useRef(false); // Prevent duplicate execution

  useEffect(() => {
    if (tryonResults.length === 0 && !hasStarted.current) {
      hasStarted.current = true;
      generateAllTryOns();
    } else if (tryonResults.length > 0) {
      setLoading(false);
    }
    // eslint-disable-next-line
  }, []);

  const generateAllTryOns = async () => {
    const selectedItems = selectedGarments.map(i => recommendations[i]);
    setProgress({ current: 0, total: selectedItems.length });
    
    const results = selectedItems.map(item => ({
      garment: item,
      result_url: null,
      processing_time: 0,
      pending: true
    }));
    
    setTryonResults(results);
    setLoading(false);
    
    console.log(`Starting ${selectedItems.length} try-ons...`);
    console.log(`Rate limit: 35s between requests`);
    console.log(`Est. time: ~${Math.ceil((selectedItems.length * 55) / 60)} mins`);
    
    for (let i = 0; i < selectedItems.length; i++) {
      const item = selectedItems[i];
      console.log(`[${i + 1}/${selectedItems.length}] Generating: ${item.name}`);
      
      try {
        const result = await generateTryOn(
          bodyPhoto.url,
          item.image_url,
          item.description,
          mapCategory(item.category)
        );
        
        if (result.success) {
          console.log(`[${i + 1}] Done in ${result.processing_time.toFixed(1)}s`);
          setTryonResults(prevResults => {
            const updated = [...prevResults];
            updated[i] = {
              garment: item,
              result_url: result.result_url,
              processing_time: result.processing_time,
              pending: false
            };
            return updated;
          });
          setProgress({ current: i + 1, total: selectedItems.length });
        } else {
          console.error(`[${i + 1}] Failed:`, result.error);
          setTryonResults(prevResults => {
            const updated = [...prevResults];
            updated[i] = {
              garment: item,
              result_url: null,
              processing_time: 0,
              pending: false,
              error: true,
              errorMessage: result.error
            };
            return updated;
          });
          setProgress({ current: i + 1, total: selectedItems.length });
        }
      } catch (error) {
        console.error(`[${i + 1}] Error:`, error);
        setTryonResults(prevResults => {
          const updated = [...prevResults];
          updated[i] = {
            garment: item,
            result_url: null,
            processing_time: 0,
            pending: false,
            error: true,
            errorMessage: error.message
          };
          return updated;
        });
        setProgress({ current: i + 1, total: selectedItems.length });
      }
      
      if (i < selectedItems.length - 1) {
        console.log(`Waiting 35s for rate limit...`);
        for (let countdown = 35; countdown > 0; countdown--) {
          if (countdown % 5 === 0 || countdown <= 3) {
            console.log(`  ${countdown}s left`);
          }
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
    }
    
    console.log(`All ${selectedItems.length} try-ons done!`);
  };

  const handleDownload = () => {
    alert('Download functionality coming soon!');
  };

  const allCompleted = tryonResults.length > 0 && tryonResults.every(r => !r.pending);

  return (
    <div className="step-content">
      <h1>YOUR VIRTUAL TRY-ONS</h1>
      <p className="subtitle">
        {allCompleted 
          ? `All ${tryonResults.length} try-ons generated!` 
          : `Generating try-ons... (${progress.current}/${progress.total} completed)`
        }
      </p>
      
      <div className="results-grid">
        {tryonResults.map((result, index) => (
          <div key={index} className="result-card">
            {result.pending ? (
              <div className="result-pending">
                {/* Blank box - no spinner */}
              </div>
            ) : result.error ? (
              <div className="result-error">
                <p>Failed</p>
              </div>
            ) : (
              <img 
                className="result-image" 
                src={`http://localhost:8003${result.result_url}`}
                alt={result.garment.name}
              />
            )}
            <div className="result-info">
              <div className="result-name">{result.garment.name}</div>
              {!result.pending && !result.error && result.result_url && (
                <div className="result-time">
                  Generated in {result.processing_time.toFixed(1)}s
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="action-buttons">
        <button className="btn-secondary" onClick={resetApp}>
          ← START OVER
        </button>
        <button 
          className="btn-primary" 
          onClick={handleDownload}
          disabled={!allCompleted}
        >
          DOWNLOAD ALL
        </button>
      </div>
    </div>
  );
};

export default ResultsStep;

