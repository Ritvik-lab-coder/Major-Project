import React, { useState } from 'react';
import axios from 'axios';

export default function PredictionForm() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [type, setType] = useState('hair');
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  };

  const handleTypeChange = (e) => {
    setType(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return alert("Please upload an image.");

    const formData = new FormData();
    formData.append("image", image);
    formData.append("type", type);

    setLoading(true);
    setPrediction('');
    try {
      const response = await axios.post("/api/predict", formData);
      setPrediction(response.data.prediction_label || "No prediction");
    } catch (err) {
      alert("Prediction failed.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center px-4">
      <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-md">
        <h1 className="text-2xl font-bold mb-4 text-center text-indigo-700">NutriScan Predictor</h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block font-semibold mb-1 text-gray-700">Select Type</label>
            <select
              className="w-full border border-gray-300 rounded-md p-2"
              value={type}
              onChange={handleTypeChange}
            >
              <option value="hair">Hair</option>
              <option value="nail">Nail</option>
              <option value="teeth">Teeth</option>
            </select>
          </div>

          <div>
            <label className="block font-semibold mb-1 text-gray-700">Upload Image</label>
            
            <div className="flex items-center space-x-4">
              <label className="cursor-pointer bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition">
                Choose Image
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>

              {image && (
                <span className="text-sm text-gray-600 truncate max-w-[120px]">
                  {image.name}
                </span>
              )}
            </div>

            {imagePreview && (
              <div className="mt-4">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="w-full h-48 object-cover rounded-md border border-gray-200"
                />
              </div>
            )}
          </div>

          <button
            type="submit"
            className="w-full bg-indigo-600 text-white py-2 rounded-md hover:bg-indigo-700 transition disabled:opacity-50"
            disabled={loading}
          >
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>

        {prediction && (
          <div className="mt-6 p-4 bg-green-100 text-green-700 rounded-lg">
            <strong>Prediction:</strong> {prediction}
          </div>
        )}
      </div>
    </div>
  );
}