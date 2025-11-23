# TIVORA React Frontend

Minimalistic, clean React app for virtual fashion try-ons.

## Design Philosophy

- **Minimalistic**: Clean, spacious, breathable UI
- **Sharp**: Square buttons, crisp borders, no rounded corners
- **Font**: Roboto Mono everywhere
- **Color Palette**:
  - `#F2EBDD` - Cream (Background)
  - `#D6A571` - Golden Tan (Primary)
  - `#C8A27E` - Medium Tan (Secondary)
  - `#6F4E37` - Deep Brown (Text)
  - `#2B1E12` - Darkest (Headers)

## Project Structure

```
src/
├── App.js                    # Main app component
├── App.css                   # Global app styles
├── index.js                  # Entry point
├── index.css                 # Reset & CSS variables
├── components/
│   ├── Header.js             # Top navigation bar
│   ├── ProgressBar.js        # Step progress indicator
│   ├── Steps.css             # Shared step styles
│   └── steps/
│       ├── LoginStep.js      # Step 1: User selection
│       ├── SkinToneStep.js   # Step 2: Skin tone analysis
│       ├── MakeupStep.js     # Step 3: Makeup try-on
│       ├── UploadStep.js     # Step 4: Body photo upload
│       ├── RecommendationsStep.js  # Step 5: AI recommendations
│       └── ResultsStep.js    # Step 6: Try-on results
├── context/
│   └── AppContext.js         # Global state management
└── services/
    └── api.js                # API integration
```

## User Flow

1. **Login** → Clerk authentication, then select gender (male/female)
2. **Skin Tone Analysis** → Upload face photo, AI detects skin tone
3. **Makeup Try-On** → Optional makeup application
4. **Body Photo** → Upload full-body standing photo
5. **Recommendations** → AI generates 15 personalized garments
6. **Try-On Results** → Select items, generate virtual try-ons

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Clerk Authentication

1. Create a Clerk account at [clerk.com](https://clerk.com)
2. Create a new application in the Clerk Dashboard
3. Go to [API Keys](https://dashboard.clerk.com/last-active?path=api-keys)
4. Copy your **Publishable Key**
5. Create `.env.local` in the `frontend/` directory:

```bash
REACT_APP_CLERK_PUBLISHABLE_KEY=pk_test_your_actual_key_here
```

**Important:** 
- Create React App uses `REACT_APP_` prefix for environment variables
- Never commit `.env.local` to Git (already in `.gitignore`)
- The app will throw an error if this key is missing

### 3. Start Backend APIs

Make sure all 3 backend APIs are running:

```bash
# Terminal 1
python backend/makeup/api.py

# Terminal 2
python backend/vto/api.py

# Terminal 3
python backend/recommendation/api.py
```

### 4. Start React App

```bash
npm start
```

App opens at `http://localhost:3000`

## Features

- ✅ **Clerk Authentication** - Secure sign-in/sign-up
- ✅ Clean, minimalistic UI
- ✅ Context-based state management
- ✅ Progressive 6-step flow
- ✅ Real-time API integration
- ✅ Image upload with preview
- ✅ Multi-select garments (max 15)
- ✅ Async try-on generation
- ✅ Fully responsive design

## API Integration

The app connects to three backend services:

- **Makeup API** (8001): Skin tone detection & makeup application
- **Recommendation API** (8002): AI-powered garment recommendations
- **VTO API** (8003): Virtual try-on generation

## Building for Production

```bash
npm run build
```

Creates optimized production build in `build/` folder.

## Tech Stack

- React 18
- **Clerk** (authentication)
- Context API (state management)
- CSS3 (no external CSS frameworks)
- Fetch API (HTTP requests)

