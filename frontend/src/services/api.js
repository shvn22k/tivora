const API_URLS = {
  MAKEUP: 'http://localhost:8001',
  RECOMMENDATION: 'http://localhost:8002',
  VTO: 'http://localhost:8003'
};
export const analyzeSkinTone = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_URLS.MAKEUP}/makeup/palettes`, {
    method: 'POST',
    body: formData
  });

  return response.json();
};

export const applyMakeup = async (imageFile, options = {}) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('apply_lipstick', options.applyLipstick ?? true);
  formData.append('apply_blush', options.applyBlush ?? true);
  formData.append('lipstick_intensity', options.lipstickIntensity ?? 0.9);
  formData.append('blush_intensity', options.blushIntensity ?? 1.0);

  const response = await fetch(`${API_URLS.MAKEUP}/makeup/apply`, {
    method: 'POST',
    body: formData
  });

  return response.blob();
};

// Recommendation API
export const getRecommendations = async (skinToneHex, gender, numItems = 6) => {
  const response = await fetch(`${API_URLS.RECOMMENDATION}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      skin_tone_hex: skinToneHex,
      gender: gender,
      num_items: numItems
    })
  });

  return response.json();
};

export const uploadPersonImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_URLS.VTO}/upload-person`, {
    method: 'POST',
    body: formData
  });

  return response.json();
};

export const generateTryOn = async (personImageUrl, garmentImageUrl, garmentDescription, category) => {
  const response = await fetch(`${API_URLS.VTO}/tryon`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      person_image_url: personImageUrl,
      garment_image_url: garmentImageUrl,
      garment_description: garmentDescription,
      category: category
    })
  });

  return response.json();
};

export const mapCategory = (category) => {
  const upperCategories = ['Tshirts', 'Shirts', 'Tops', 'Tunics', 'Sweatshirts', 'Jackets'];
  const lowerCategories = ['Jeans', 'Trousers', 'Shorts', 'Track Pants'];
  
  if (upperCategories.includes(category)) return 'upper_body';
  if (lowerCategories.includes(category)) return 'lower_body';
  return 'upper_body';
};

export const detectUndertone = (hex) => {
  const r = parseInt(hex.substr(1, 2), 16);
  const g = parseInt(hex.substr(3, 2), 16);
  const b = parseInt(hex.substr(5, 2), 16);
  
  if (r > g && r > b) {
    return (r - g) > 15 ? 'Warm' : 'Neutral';
  } else if (g > r && g > b) {
    return 'Cool';
  } else if (b > g) {
    return 'Cool';
  }
  return 'Neutral';
};

