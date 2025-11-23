import React from 'react';
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react';
import { useApp } from '../context/AppContext';

const Header = () => {
  const { user } = useApp();

  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">TIVORA</div>
        <div className="user-info">
          <SignedIn>
            <span>{user.name || 'User'}</span>
            <UserButton 
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  avatarBox: {
                    width: '36px',
                    height: '36px'
                  }
                }
              }}
            />
          </SignedIn>
          <SignedOut>
            <span>Guest</span>
          </SignedOut>
        </div>
      </div>
    </header>
  );
};

export default Header;

