import React, { useEffect, useState } from 'react';
import { useUser, SignInButton, SignUpButton } from '@clerk/clerk-react';
import { useApp } from '../../context/AppContext';
import '../Steps.css';

const LoginStep = () => {
  const { user: clerkUser } = useUser();
  const { setUser, nextStep } = useApp();
  const [selectedGender, setSelectedGender] = useState('');
  const [showGenderSelection, setShowGenderSelection] = useState(false);

  useEffect(() => {
    if (clerkUser) {
      // User is signed in, show gender selection
      setShowGenderSelection(true);
    }
  }, [clerkUser]);

  const handleSelectGender = (gender) => {
    setSelectedGender(gender);
    setUser({ 
      name: clerkUser?.fullName || clerkUser?.firstName || 'User',
      gender 
    });
    console.log(`User selected: ${gender}`);
    setTimeout(() => nextStep(), 300);
  };

  if (!clerkUser) {
    return (
      <div className="step-content">
        <h1>WELCOME TO TIVORA</h1>
        <p className="subtitle">Your AI-Powered Virtual Styling Studio</p>
        
        <div className="card">
          <h2>Sign In to Continue</h2>
          <p style={{ marginBottom: 'var(--spacing-lg)', color: 'var(--deep-brown)' }}>
            Create an account or sign in to access your personalized virtual styling experience.
          </p>
          <div className="choice-buttons">
            <SignInButton mode="modal">
              <button className="btn-choice">
                <span className="choice-text">Sign In</span>
              </button>
            </SignInButton>
            <SignUpButton mode="modal">
              <button className="btn-choice">
                <span className="choice-text">Sign Up</span>
              </button>
            </SignUpButton>
          </div>
        </div>
      </div>
    );
  }

  if (showGenderSelection) {
    return (
      <div className="step-content">
        <h1>WELCOME, {clerkUser.firstName?.toUpperCase()}!</h1>
        <p className="subtitle">Select your gender to get personalized recommendations</p>
        
        <div className="card">
          <h2>Select Your Gender</h2>
          <div className="profile-grid">
            <div 
              className="profile-card" 
              onClick={() => handleSelectGender('male')}
            >
              <div className="profile-name">Male</div>
              <div className="profile-gender">Men's Fashion</div>
            </div>
            <div 
              className="profile-card" 
              onClick={() => handleSelectGender('female')}
            >
              <div className="profile-name">Female</div>
              <div className="profile-gender">Women's Fashion</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default LoginStep;

