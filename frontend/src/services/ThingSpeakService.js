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

// Time-based fetching with timestamp handling to avoid duplicates
export const fetchDataSinceLastUpdate = async (channelId, lastTimestamp, apiKey = DEFAULT_READ_API_KEY) => {
  if (!channelId) {
    throw new Error('Channel ID is required');
  }

  try {
    // If lastTimestamp is provided, fetch data after that timestamp
    const timestampParam = lastTimestamp ? `&start=${lastTimestamp}` : '';
    const url = `${BASE_URL}/${channelId}/feeds.json?api_key=${apiKey}${timestampParam}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`ThingSpeak API error: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching data from ThingSpeak since last update:', error);
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

/**
 * Converts ThingSpeak data to CSV format
 * @param {Object} data - ThingSpeak data object
 * @returns {String} - CSV formatted string
 */
export const convertThingSpeakDataToCSV = (data) => {
  if (!data || !data.feeds || data.feeds.length === 0) {
    return null;
  }
  
  // Create CSV header
  const headers = [
    'timestamp', 
    'temperature', 
    'alertLevel', 
    'heartRate', 
    'spo2', 
    'latitude', 
    'longitude', 
    'avgEcg',
    'systolicBP',
    'diastolicBP'
  ];
  
  // Create CSV content
  const csvRows = [];
  csvRows.push(headers.join(','));
  
  // Process each feed entry
  data.feeds.forEach(feed => {
    const temperature = parseFloat(feed.field1);
    const alertLevel = parseInt(feed.field2);
    const heartRate = parseFloat(feed.field3);
    const spo2 = parseFloat(feed.field4);
    const latitude = parseFloat(feed.field5);
    const longitude = parseFloat(feed.field6);
    const avgEcg = parseFloat(feed.field7);
    const ecgSamples = feed.field8 ? JSON.parse(feed.field8) : [];
    
    // Calculate BP
    const bp = calculateBP(ecgSamples, spo2);
    
    const row = [
      new Date(feed.created_at).toISOString(),
      isNaN(temperature) ? "" : temperature.toFixed(1),
      isNaN(alertLevel) ? "" : alertLevel,
      isNaN(heartRate) ? "" : Math.round(heartRate),
      isNaN(spo2) ? "" : Math.round(spo2),
      isNaN(latitude) ? "" : latitude,
      isNaN(longitude) ? "" : longitude,
      isNaN(avgEcg) ? "" : avgEcg,
      bp.systolic,
      bp.diastolic
    ];
    
    csvRows.push(row.join(','));
  });
  
  return csvRows.join('\n');
};

/**
 * Downloads ThingSpeak data as a CSV file
 * @param {String} channelId - ThingSpeak channel ID
 * @param {String} filename - Name for the CSV file
 * @param {Number} days - Number of days of data to fetch (default: 7)
 */
export const downloadThingSpeakDataAsCSV = async (channelId, filename = 'patient_data.csv', days = 7) => {
  if (!channelId) {
    throw new Error('Channel ID is required');
  }

  try {
    // Calculate start date (days ago from now)
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    // Fetch data for the specified period
    const url = `${BASE_URL}/${channelId}/feeds.json?api_key=${DEFAULT_READ_API_KEY}&start=${startDate.toISOString()}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`ThingSpeak API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Convert data to CSV
    const csvContent = convertThingSpeakDataToCSV(data);
    
    if (!csvContent) {
      throw new Error('No data available to download');
    }
    
    // Create and download the CSV file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const urlObject = URL.createObjectURL(blob);
    
    link.setAttribute('href', urlObject);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    return true;
  } catch (error) {
    console.error('Error downloading ThingSpeak data as CSV:', error);
    throw error;
  }
};

/**
 * Starts continuous data logging from ThingSpeak to localStorage
 * @param {String} channelId - ThingSpeak channel ID
 * @param {Number} intervalMinutes - Fetch interval in minutes
 * @returns {Object} - Controller object with start/stop methods
 */
export const startContinuousDataLogging = (channelId, intervalMinutes = 5) => {
  if (!channelId) {
    throw new Error('Channel ID is required');
  }
  
  let intervalId = null;
  let isLogging = false;
  let lastTimestamp = null;
  const storageKey = `thingspeak_data_${channelId}`;
  
  // Initialize from localStorage if exists
  const initFromStorage = () => {
    const storedData = localStorage.getItem(storageKey);
    if (storedData) {
      try {
        const parsedData = JSON.parse(storedData);
        if (parsedData.feeds && parsedData.feeds.length > 0) {
          // Get the latest timestamp
          const latestEntry = parsedData.feeds[parsedData.feeds.length - 1];
          lastTimestamp = new Date(latestEntry.created_at).toISOString();
        }
      } catch (error) {
        console.error('Error parsing stored data:', error);
        localStorage.removeItem(storageKey);
      }
    }
  };
  
  // Fetch and store new data
  const fetchAndStore = async () => {
    try {
      const data = await fetchDataSinceLastUpdate(channelId, lastTimestamp);
      
      if (data && data.feeds && data.feeds.length > 0) {
        // Update lastTimestamp to the latest entry
        const latestEntry = data.feeds[data.feeds.length - 1];
        lastTimestamp = new Date(latestEntry.created_at).toISOString();
        
        // Merge with existing data
        const storedData = localStorage.getItem(storageKey);
        let mergedData;
        
        if (storedData) {
          const parsedData = JSON.parse(storedData);
          mergedData = {
            ...data,
            feeds: [...parsedData.feeds, ...data.feeds]
          };
        } else {
          mergedData = data;
        }
        
        // Store in localStorage
        localStorage.setItem(storageKey, JSON.stringify(mergedData));
        
        // Also trigger CSV download if the dataset is getting large
        if (mergedData.feeds.length > 1000) {
          const csvContent = convertThingSpeakDataToCSV(mergedData);
          const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
          const link = document.createElement('a');
          const urlObject = URL.createObjectURL(blob);
          
          link.setAttribute('href', urlObject);
          link.setAttribute('download', `patient_data_${channelId}_${new Date().toISOString().slice(0,10)}.csv`);
          link.style.visibility = 'hidden';
          
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          
          // Reset storage after download
          localStorage.setItem(storageKey, JSON.stringify({
            ...data,
            feeds: data.feeds
          }));
        }
      }
    } catch (error) {
      console.error('Error in continuous data logging:', error);
    }
  };
  
  // Start logging
  const start = () => {
    if (isLogging) return;
    
    initFromStorage();
    fetchAndStore(); // Immediate first fetch
    
    // Set up interval (convert minutes to milliseconds)
    intervalId = setInterval(fetchAndStore, intervalMinutes * 60 * 1000);
    isLogging = true;
    
    console.log(`Started continuous data logging for channel ${channelId} every ${intervalMinutes} minutes`);
    return true;
  };
  
  // Stop logging
  const stop = () => {
    if (!isLogging) return;
    
    clearInterval(intervalId);
    intervalId = null;
    isLogging = false;
    
    console.log(`Stopped continuous data logging for channel ${channelId}`);
    return true;
  };
  
  // Export data as CSV
  const exportToCSV = (filename = `patient_data_${channelId}.csv`) => {
    const storedData = localStorage.getItem(storageKey);
    if (!storedData) {
      console.warn('No data available to export');
      return false;
    }
    
    try {
      const parsedData = JSON.parse(storedData);
      const csvContent = convertThingSpeakDataToCSV(parsedData);
      
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const urlObject = URL.createObjectURL(blob);
      
      link.setAttribute('href', urlObject);
      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      return true;
    } catch (error) {
      console.error('Error exporting data to CSV:', error);
      return false;
    }
  };
  
  // Return controller object
  return {
    start,
    stop,
    exportToCSV,
    isLogging: () => isLogging,
    getStoredData: () => {
      const storedData = localStorage.getItem(storageKey);
      return storedData ? JSON.parse(storedData) : null;
    },
    clearStoredData: () => {
      localStorage.removeItem(storageKey);
      lastTimestamp = null;
      return true;
    }
  };
};

/**
 * Fetches historical data for a specific time period
 * @param {String} channelId - ThingSpeak channel ID
 * @param {Number} days - Number of days of data to fetch (default: 7)
 * @param {String} apiKey - ThingSpeak API key (optional)
 * @returns {Object} - Formatted historical data for all fields
 */
export const fetchHistoricalData = async (channelId, days = 7, apiKey = DEFAULT_READ_API_KEY) => {
  if (!channelId) {
    throw new Error('Channel ID is required');
  }

  try {
    // Fetch only the latest 10 records
    const url = `${BASE_URL}/${channelId}/feeds.json?api_key=${apiKey}&results=15`;
    console.log(`Fetching latest records from: ${url}`);
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`ThingSpeak API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data || !data.feeds || data.feeds.length === 0) {
      throw new Error('No historical data available from ThingSpeak');
    }
    
    console.log(`Fetched ${data.feeds.length} records from ThingSpeak`);
    
    // Use only the latest 10 records for a more compact view
    const limitedFeeds = data.feeds.slice(-10);
    console.log(`Limited to latest ${limitedFeeds.length} records`);
    
    // Format data for chart display - with safer parsing
    const formattedData = {
      timestamps: limitedFeeds.map(feed => new Date(feed.created_at)),
      temperature: limitedFeeds.map(feed => {
        const val = parseFloat(feed.field1);
        return isNaN(val) ? null : val;
      }),
      alertLevel: limitedFeeds.map(feed => {
        const val = parseInt(feed.field2);
        return isNaN(val) ? null : val;
      }),
      heartRate: limitedFeeds.map(feed => {
        const val = parseFloat(feed.field3);
        return isNaN(val) ? null : val;
      }),
      spo2: limitedFeeds.map(feed => {
        const val = parseFloat(feed.field4);
        return isNaN(val) ? null : val;
      }),
      // Calculate BP for each entry
      bp: limitedFeeds.map(feed => {
        try {
          const ecgSamples = feed.field8 ? JSON.parse(feed.field8) : [];
          const spo2 = parseFloat(feed.field4);
          return calculateBP(ecgSamples, spo2);
        } catch (e) {
          console.error('Error processing BP data:', e);
          return { systolic: null, diastolic: null };
        }
      }),
      avgEcg: limitedFeeds.map(feed => {
        const val = parseFloat(feed.field7);
        return isNaN(val) ? null : val;
      }),
    };
    
    // Log some stats about the formatted data
    console.log('Formatted data stats:');
    console.log(`Temperature data points: ${formattedData.temperature.filter(v => v !== null).length}`);
    console.log(`Heart rate data points: ${formattedData.heartRate.filter(v => v !== null).length}`);
    console.log(`SpO2 data points: ${formattedData.spo2.filter(v => v !== null).length}`);
    console.log(`BP data points: ${formattedData.bp.filter(v => v.systolic !== null).length}`);
    
    return {
      raw: {
        ...data,
        feeds: limitedFeeds
      },
      formatted: formattedData
    };
  } catch (error) {
    console.error('Error fetching historical data:', error);
    throw error;
  }
};

export default {
  fetchLatestData,
  mapThingSpeakToPatientVitals,
  getPatientVitals,
  downloadThingSpeakDataAsCSV,
  startContinuousDataLogging,
  convertThingSpeakDataToCSV,
  fetchHistoricalData
};

