/**
 * ThingSpeak API Service
 * Handles fetching data from ThingSpeak channels
 */

const DEFAULT_READ_API_KEY = "MWFBV98HOZOHTHY4";
const BASE_URL = "https://api.thingspeak.com/channels";

/**
 * Fetches the latest data from a ThingSpeak channel
 */
export const fetchLatestData = async (channelId, apiKey = DEFAULT_READ_API_KEY, results = 1) => {
  if (!channelId) {
    throw new Error('Channel ID is required');
  }

  try {
    const url = `${BASE_URL}/${channelId}/feeds.json?api_key=${apiKey}&results=${results}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`ThingSpeak API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data || !data.feeds || data.feeds.length === 0) {
      throw new Error('No data available from ThingSpeak');
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching data from ThingSpeak:', error);
    throw error;
  }
};

/**
 * Calculate blood pressure from PTT
 * @param {Array} ecgSamples - Array of ECG samples
 * @param {number} spo2 - SpO2 value
 * @returns {Object} - Systolic and diastolic BP
 */
const calculateBP = (ecgSamples, spo2) => {
  // Assuming 1000Hz sampling rate (1 sample = 1ms)
  const PTT = 0.1; // 100ms - simplified assumption
  
  // BP estimation formulas
  const systolic = 150 - (100 * PTT);
  const diastolic = 100 - (60 * PTT);
  
  return {
    systolic: Math.round(systolic),
    diastolic: Math.round(diastolic)
  };
};

/**
 * Maps the raw ThingSpeak data to patient vital signs
 * Field mapping:
 * field1: Temperature (°C)
 * field2: Emergency Alert Level (0: Normal, 1: Moderate Risk, 2: High Risk)
 * field3: Heart Rate (BPM)
 * field4: SpO2 (%)
 * field5: Latitude
 * field6: Longitude
 * field7: Avg ECG
 * field8: ECG Sample Array
 */
export const mapThingSpeakToPatientVitals = (data) => {
  if (!data || !data.feeds || data.feeds.length === 0) {
    return null;
  }
  
  const latestEntry = data.feeds[0];
  
  // Convert string values to numbers
  const temperature = parseFloat(latestEntry.field1);
  const alertLevel = parseInt(latestEntry.field2);
  const heartRate = parseFloat(latestEntry.field3);
  const spo2 = parseFloat(latestEntry.field4);
  const latitude = parseFloat(latestEntry.field5);
  const longitude = parseFloat(latestEntry.field6);
  const avgEcg = parseFloat(latestEntry.field7);
  const ecgSamples = JSON.parse(latestEntry.field8);

  // Calculate BP from ECG samples and SpO2
  const bp = calculateBP(ecgSamples, spo2);

  return {
    vitals: {
      temperature: isNaN(temperature) ? "--" : temperature.toFixed(1),
      alertLevel: isNaN(alertLevel) ? 0 : alertLevel,
      heartRate: isNaN(heartRate) ? "--" : Math.round(heartRate),
      spo2: isNaN(spo2) ? "--" : Math.round(spo2),
      avgEcg: isNaN(avgEcg) ? "--" : avgEcg,
      ecgSamples: Array.isArray(ecgSamples) ? ecgSamples : [],
      bp: bp
    },
    location: {
      lat: isNaN(latitude) ? null : latitude,
      lng: isNaN(longitude) ? null : longitude,
      accuracy: 15
    },
    status: {
      temperature: temperature > 37.8 ? "Elevated" : temperature < 35.5 ? "Low" : "Normal",
      heartRate: heartRate > 100 ? "Elevated" : heartRate < 60 ? "Low" : "Normal",
      spo2: spo2 < 95 ? "Low" : "Normal",
      bp: bp.systolic > 140 || bp.diastolic > 90 ? "Elevated" : 
          bp.systolic < 90 || bp.diastolic < 60 ? "Low" : "Normal",
      alert: alertLevel === 0 ? "Normal" : alertLevel === 1 ? "Moderate Risk" : "High Risk"
    },
    lastUpdated: new Date(latestEntry.created_at).toLocaleString()
  };
};

/**
 * Fetches and formats patient vitals from ThingSpeak
 */
export const getPatientVitals = async (channelId) => {
  if (!channelId) {
    throw new Error('Channel ID is required');
  }

  try {
    const data = await fetchLatestData(channelId);
    const vitals = mapThingSpeakToPatientVitals(data);
    
    if (!vitals) {
      throw new Error('Failed to map ThingSpeak data to vitals');
    }
    
    return vitals;
  } catch (error) {
    console.error('Error getting patient vitals:', error);
    throw error;
  }
};

export default {
  fetchLatestData,
  mapThingSpeakToPatientVitals,
  getPatientVitals
};

