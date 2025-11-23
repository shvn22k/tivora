import React from 'react';
import { AppProvider, useApp } from './context/AppContext';
import Header from './components/Header';
import ProgressBar from './components/ProgressBar';
import LoginStep from './components/steps/LoginStep';
import SkinToneStep from './components/steps/SkinToneStep';
import MakeupStep from './components/steps/MakeupStep';
import UploadStep from './components/steps/UploadStep';
import RecommendationsStep from './components/steps/RecommendationsStep';
import ResultsStep from './components/steps/ResultsStep';
import './App.css';

const AppContent = () => {
  const { currentStep } = useApp();

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return <LoginStep />;
      case 2:
        return <SkinToneStep />;
      case 3:
        return <MakeupStep />;
      case 4:
        return <UploadStep />;
      case 5:
        return <RecommendationsStep />;
      case 6:
        return <ResultsStep />;
      default:
        return <LoginStep />;
    }
  };

  return (
    <div className="app">
      <Header />
      <ProgressBar />
      <main className="main-container">
        <div className="step-section">
          {renderStep()}
        </div>
      </main>
    </div>
  );
};

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;

