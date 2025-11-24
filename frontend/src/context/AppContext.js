import React, { createContext, useContext, useState } from 'react';

const AppContext = createContext();

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};

export const AppProvider = ({ children }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [user, setUser] = useState({ name: '', gender: '' });
  const [skinTone, setSkinTone] = useState({ hex: '', undertone: '', image: null });
  const [makeup, setMakeup] = useState({ applied: false, image: null });
  const [bodyPhoto, setBodyPhoto] = useState({ image: null, url: '', previewUrl: '' });
  const [recommendations, setRecommendations] = useState([]);
  const [selectedGarments, setSelectedGarments] = useState([]);
  const [tryonResults, setTryonResults] = useState([]);

  const nextStep = () => {
    if (currentStep < 6) {
      setCurrentStep(currentStep + 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const previousStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const resetApp = () => {
    setCurrentStep(1);
    setUser({ name: '', gender: '' });
    setSkinTone({ hex: '', undertone: '', image: null });
    setMakeup({ applied: false, image: null });
    setBodyPhoto({ image: null, url: '', previewUrl: '' });
    setRecommendations([]);
    setSelectedGarments([]);
    setTryonResults([]);
  };

  const toggleGarmentSelection = (index) => {
    if (selectedGarments.includes(index)) {
      setSelectedGarments(selectedGarments.filter(i => i !== index));
    } else {
      if (selectedGarments.length >= 6) {
        alert('Maximum 6 garments can be selected');
        return;
      }
      setSelectedGarments([...selectedGarments, index]);
    }
  };

  const value = {
    currentStep,
    setCurrentStep,
    user,
    setUser,
    skinTone,
    setSkinTone,
    makeup,
    setMakeup,
    bodyPhoto,
    setBodyPhoto,
    recommendations,
    setRecommendations,
    selectedGarments,
    setSelectedGarments,
    tryonResults,
    setTryonResults,
    nextStep,
    previousStep,
    resetApp,
    toggleGarmentSelection
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

