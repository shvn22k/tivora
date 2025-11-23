import React, { useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { getRecommendations } from '../../services/api';
import '../Steps.css';

const RecommendationsStep = () => {
  const {  
    user,
    skinTone,
    recommendations,
    setRecommendations,
    setSelectedGarments,
    nextStep
  } = useApp();
  
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRecommendations();
    // eslint-disable-next-line
  }, []);

  const loadRecommendations = async () => {
    setLoading(true);
    try {
      const data = await getRecommendations(skinTone.hex, user.gender, 15);
      setRecommendations(data);
      console.log(`Loaded ${data.length} recommendations`);
      
      const allIndices = Array.from({ length: data.length }, (_, i) => i);
      setSelectedGarments(allIndices);
      console.log(`Selected ${data.length} items`);
      
      setTimeout(() => {
        nextStep();
      }, 1500);
      
    } catch (error) {
      console.error('Failed to load recommendations:', error);
      alert('Failed to load recommendations. Please try again.');
      setLoading(false);
    }
  };

  return (
    <div className="step-content">
      <h1>YOUR RECOMMENDATIONS</h1>
      <p className="subtitle">AI-curated garments matching your skin tone and style</p>
      
      <div className="loading">
        <div className="spinner"></div>
        <p>{recommendations.length > 0 
          ? 'Recommendations generated! Preparing try-ons...' 
          : 'Generating personalized recommendations...'
        }</p>
      </div>
    </div>
  );
};

export default RecommendationsStep;

