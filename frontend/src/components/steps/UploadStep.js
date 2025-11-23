import React, { useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { uploadPersonImage } from '../../services/api';
import '../Steps.css';

const UploadStep = () => {
  const { bodyPhoto, setBodyPhoto, nextStep, previousStep } = useApp();
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);

  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    try {
      const result = await uploadPersonImage(file);
      if (result.success) {
        const blobUrl = URL.createObjectURL(file);
        setBodyPhoto({ 
          image: file, 
          url: result.image_url,
          previewUrl: blobUrl
        });
        console.log('Uploaded:', result.image_url);
      } else {
        alert('Failed to upload image. Please try again.');
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload image. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="step-content">
      <h1>UPLOAD YOUR PHOTO</h1>
      <p className="subtitle">Upload a full-body standing photo for accurate virtual try-ons</p>
      
      <div className="card">
        {!bodyPhoto.url ? (
          <div 
            className="upload-zone" 
            onClick={() => fileInputRef.current?.click()}
          >
            <p>Click to upload or drag and drop</p>
            <span className="upload-hint">PNG, JPG up to 10MB • Stand straight, neutral background</span>
            <input 
              ref={fileInputRef}
              type="file" 
              accept="image/*" 
              hidden 
              onChange={handleFileSelect}
            />
          </div>
        ) : (
          <div className="preview-container">
            <img src={bodyPhoto.previewUrl || bodyPhoto.url} alt="Body Preview" />
            {uploading && <p style={{ marginTop: '10px', color: '#6F4E37' }}>Uploading...</p>}
          </div>
        )}
      </div>

      <div className="action-buttons">
        <button className="btn-secondary" onClick={previousStep} disabled={uploading}>
          ← BACK
        </button>
        <button 
          className="btn-primary" 
          onClick={nextStep}
          disabled={!bodyPhoto.url || uploading}
        >
          {uploading ? 'UPLOADING...' : 'CONTINUE →'}
        </button>
      </div>
    </div>
  );
};

export default UploadStep;

