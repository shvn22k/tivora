// Simple Tivora Webstore

let user = null;
let recommendations = [];

// Show/hide steps
function showStep(stepId) {
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    document.getElementById(stepId).classList.add('active');
}

// Login
async function login(gender) {
    user = { gender, id: `user_${Date.now()}` };
    
    if (gender === 'female') {
        showStep('step-skintone');
    } else {
        showStep('step-photo');
    }
}

// Detect skin tone
async function detectSkinTone() {
    const file = document.getElementById('face-photo').files[0];
    if (!file) {
        alert('Select a photo');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const res = await fetch('http://localhost:8001/detect-skin-tone', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        document.getElementById('skin-result').textContent = `Skin tone: ${data.skin_tone}`;
        document.getElementById('skin-result').classList.remove('hidden');
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

function skipSkinTone() {
    showStep('step-photo');
}

// Get recommendations
async function getRecommendations() {
    const file = document.getElementById('body-photo').files[0];
    if (!file) {
        alert('Upload your photo');
        return;
    }
    
    // Store photo for later
    user.photo = file;
    
    try {
        const res = await fetch('http://localhost:8002/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gender: user.gender, num_items: 15 })
        });
        recommendations = (await res.json()).garments;
        
        // Display recommendations
        const grid = document.getElementById('rec-grid');
        grid.innerHTML = recommendations.map(g => `
            <div class="border p-4 rounded">
                <div class="bg-gray-200 h-40 mb-2"></div>
                <p class="text-sm font-bold">${g.name.substring(0, 30)}...</p>
                <p class="text-xs text-gray-600">${g.category}</p>
            </div>
        `).join('');
        
        showStep('step-recommendations');
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// Generate try-ons
async function generateTryOns() {
    showStep('step-results');
    document.getElementById('results-loading').classList.remove('hidden');
    
    // Convert photo to base64
    const reader = new FileReader();
    reader.onload = async function(e) {
        const userImageUrl = e.target.result;
        
        const results = [];
        for (let garment of recommendations.slice(0, 10)) {
            try {
                const res = await fetch('http://localhost:8003/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_image_url: userImageUrl,
                        garment_image_url: garment.image_url,
                        garment_description: garment.description,
                        category: "upper_body"
                    })
                });
                const data = await res.json();
                results.push({ ...data, garment });
            } catch (e) {
                results.push({ success: false, error: e.message, garment });
            }
        }
        
        // Show results
        document.getElementById('results-loading').classList.add('hidden');
        document.getElementById('results-content').classList.remove('hidden');
        
        const grid = document.getElementById('results-grid');
        grid.innerHTML = results.map(r => `
            <div class="border p-4 rounded ${r.success ? 'bg-green-50' : 'bg-red-50'}">
                <div class="bg-gray-200 h-64 mb-2"></div>
                <p class="text-sm font-bold">${r.garment.name.substring(0, 30)}</p>
                <p class="text-xs ${r.success ? 'text-green-600' : 'text-red-600'}">
                    ${r.success ? '✓ Generated!' : '✗ Failed'}
                </p>
            </div>
        `).join('');
    };
    reader.readAsDataURL(user.photo);
}
