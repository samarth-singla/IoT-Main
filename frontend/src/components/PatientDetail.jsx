import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { getPatientVitals } from '../services/ThingSpeakService';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
import 'leaflet/dist/leaflet.css';
import './PatientDetail.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

const PatientDetail = () => {
  const { id } = useParams();
  const [patientData, setPatientData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ECG Chart configuration
  const ecgChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'ECG Waveform'
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Sample'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'mV'
        }
      }
    },
    animation: false // Disable animation for better performance
  };

  const prepareEcgChartData = (samples) => ({
    labels: samples.map((_, index) => index),
    datasets: [
      {
        label: 'ECG',
        data: samples,
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.1
      }
    ]
  });

  const fetchVitals = useCallback(async () => {
    if (!id) return;

    try {
      const vitalsData = await getPatientVitals(id);
      setPatientData(prev => ({
        ...prev,
        ...vitalsData
      }));
      setLoading(false);
    } catch (error) {
      console.error('Error fetching vitals:', error);
      setError(error.message);
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    fetchVitals();
    const interval = setInterval(fetchVitals, 30000);
    return () => clearInterval(interval);
  }, [fetchVitals]);

  if (loading) return <div className="loading">Loading patient data...</div>;
  if (error) return <div className="error"><h3>Error loading patient data</h3><p>{error}</p></div>;
  if (!patientData) return <div className="error">No patient data available</div>;

  // Helper function to determine temperature status
  const getTemperatureStatus = (temp) => {
    if (temp > 37.8) return 'elevated';
    if (temp < 35.5) return 'low';
    return 'normal';
  };

  // Helper function to determine heart rate status
  const getHeartRateStatus = (hr) => {
    if (hr > 100) return 'elevated';
    if (hr < 60) return 'low';
    return 'normal';
  };

  return (
    <div className="patient-detail">
      <div className="vital-signs">
        <h2>Vital Signs</h2>
        <div className="vitals-grid">
          <div className="vital-card">
            <h3>Temperature</h3>
            <p className={`value ${getTemperatureStatus(patientData.vitals.temperature)}`}>
              {patientData.vitals.temperature}°C
            </p>
          </div>
          
          <div className="vital-card">
            <h3>Humidity</h3>
            <p className="value">
              {patientData.vitals.humidity}%
            </p>
          </div>
          
          <div className="vital-card">
            <h3>Heart Rate</h3>
            <p className={`value ${getHeartRateStatus(patientData.vitals.heartRate)}`}>
              {patientData.vitals.heartRate} BPM
            </p>
          </div>
          
          <div className="vital-card">
            <h3>SpO2</h3>
            <p className={`value ${patientData.vitals.spo2 < 95 ? 'low' : 'normal'}`}>
              {patientData.vitals.spo2}%
            </p>
          </div>

          <div className="vital-card">
            <h3>Blood Pressure</h3>
            <p className={`value ${patientData.status.bp}`}>
              {patientData.vitals.bp.systolic}/{patientData.vitals.bp.diastolic} mmHg
            </p>
          </div>

          <div className="vital-card">
            <h3>Average ECG</h3>
            <p className="value">
              {patientData.vitals.avgEcg.toFixed(2)} mV
            </p>
          </div>

          {patientData.location.lat && patientData.location.lng && (
            <div className="vital-card location-card">
              <h3>Location</h3>
              <p>Lat: {patientData.location.lat}</p>
              <p>Long: {patientData.location.lng}</p>
            </div>
          )}
        </div>
      </div>

      {/* ECG Graph Section */}
      <div className="ecg-section">
        <h2>ECG Waveform</h2>
        <div className="ecg-chart-container">
          {patientData.vitals.ecgSamples.length > 0 ? (
            <Line
              options={ecgChartOptions}
              data={prepareEcgChartData(patientData.vitals.ecgSamples)}
            />
          ) : (
            <p>No ECG data available</p>
          )}
        </div>
      </div>

      {/* Map Section */}
      {patientData.location.lat && patientData.location.lng && (
        <div className="map-section">
          <h2>Patient Location</h2>
          <div className="map-container">
            <MapContainer
              center={[patientData.location.lat, patientData.location.lng]}
              zoom={13}
              style={{ height: '400px', width: '100%' }}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              <Marker position={[patientData.location.lat, patientData.location.lng]}>
                <Popup>
                  Patient's Location<br />
                  Lat: {patientData.location.lat}<br />
                  Long: {patientData.location.lng}
                </Popup>
              </Marker>
            </MapContainer>
          </div>
        </div>
      )}

      <div className="update-info">
        <p>Last Updated: {patientData.lastUpdated}</p>
      </div>
    </div>
  );
};

export default PatientDetail;

