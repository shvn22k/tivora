# Clerk Authentication Setup

## Quick Setup Guide

### 1. Create Clerk Account

Visit [https://clerk.com](https://clerk.com) and sign up for a free account.

### 2. Create Application

1. In the Clerk Dashboard, click **"+ Create Application"**
2. Name it "TIVORA" or any name you prefer
3. Choose your sign-in options (Email, Google, etc.)
4. Click **"Create Application"**

### 3. Get Your Publishable Key

1. Go to **API Keys** page: [https://dashboard.clerk.com/last-active?path=api-keys](https://dashboard.clerk.com/last-active?path=api-keys)
2. Select **"React"** from the framework dropdown
3. Copy your **Publishable Key** (starts with `pk_test_` for development)

### 4. Configure Environment Variable

Create a file named `.env.local` in the `frontend/` directory:

```bash
REACT_APP_CLERK_PUBLISHABLE_KEY=pk_test_your_actual_key_here
```

**Replace** `pk_test_your_actual_key_here` with your actual Publishable Key from step 3.

### 5. Install Dependencies

```bash
cd frontend
npm install
```

This will install `@clerk/clerk-react` along with other dependencies.

### 6. Start the App

```bash
npm start
```

The app will:
- Check for the `REACT_APP_CLERK_PUBLISHABLE_KEY`
- Throw an error if missing
- Open at `http://localhost:3000` if configured correctly

### 7. Test Authentication

1. Click **"Sign Up"** to create a test account
2. Complete the sign-up flow
3. Select your gender (Male/Female)
4. Continue through the app flow

## Important Notes

- **Never commit** `.env.local` to Git (it's already in `.gitignore`)
- The `REACT_APP_` prefix is required for Create React App
- Use **Publishable Key**, not Secret Key (secret keys are for backend only)
- In production, use your production Publishable Key (starts with `pk_live_`)

## Troubleshooting

### Error: "Missing Clerk Publishable Key"

**Solution:** Create `.env.local` with the correct environment variable name:
```bash
REACT_APP_CLERK_PUBLISHABLE_KEY=your_key_here
```

### Sign-in modal not appearing

**Solution:** Check browser console for errors. Make sure:
1. Clerk is properly configured
2. Your publishable key is valid
3. You're using the correct key for your environment

### User not staying signed in

**Solution:** Clerk uses cookies for session management. Make sure:
1. Your browser allows cookies
2. You're not in incognito/private mode (or Clerk is configured for it)

## Additional Resources

- [Clerk React Quickstart](https://clerk.com/docs/quickstarts/react)
- [Clerk React SDK Reference](https://clerk.com/docs/references/react/overview)
- [Clerk Dashboard](https://dashboard.clerk.com/)

