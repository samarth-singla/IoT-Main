import React, { useState, useEffect } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import './PatientDetail.css';

const PatientDetail = () => {
  const { id } = useParams();
  const location = useLocation();
  const [patient, setPatient] = useState(null);
  const [vitals, setVitals] = useState({
    heartRate: 0,
    spo2: 0,
    temperature: 0,
    ecgData: {
      labels: [],
      datasets: [{
        label: 'ECG Reading',
        data: [],
        borderColor: '#4CAF50',
        tension: 0.1
      }]
    }
  });
  const [location, setLocation] = useState({ lat: 51.505, lng: -0.09 });

  // Fetch patient details
  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        // If we have state data, use it
        if (location.state?.patientData) {
          setPatient(location.state.patientData);
          return;
        }

        // Otherwise fetch from API
        const response = await fetch(`http://localhost:8000/patient/${id}`);
        if (!response.ok) {
          throw new Error('Failed to fetch patient details');
        }
        const data = await response.json();
        setPatient(data);
      } catch (error) {
        console.error("Error fetching patient data:", error);
      }
    };

    fetchPatientData();
  }, [id, location.state]);

  // Simulate real-time data updates
  useEffect(() => {
    if (!patient) return;
    
    const socket = new WebSocket('wss://mock-server/patient-vitals');
    
    // In a real app, this would be actual WebSocket events
    // Here we're simulating with setInterval
    const interval = setInterval(() => {
      // Simulate heart rate between 60-140
      const newHeartRate = Math.floor(Math.random() * 80) + 60;
      
      // Simulate SpO2 between 88-100
      const newSpo2 = Math.floor(Math.random() * 12) + 88;
      
      // Simulate temperature between 36-39
      const newTemp = (Math.random() * 3 + 36).toFixed(1);
      
      // Simulate ECG data
      const labels = vitals.ecgData.labels.slice(-19);
      labels.push(new Date().toLocaleTimeString());
      
      const ecgValues = vitals.ecgData.datasets[0].data.slice(-19);
      // Generate a sine wave-like pattern for ECG
      const lastValue = ecgValues.length > 0 ? ecgValues[ecgValues.length - 1] : 0;
      const newValue = lastValue + (Math.random() * 2 - 1) * 0.5;
      ecgValues.push(newValue);
      
      setVitals({
        heartRate: newHeartRate,
        spo2: newSpo2,
        temperature: newTemp,
        ecgData: {
          labels: labels,
          datasets: [{
            ...vitals.ecgData.datasets[0],
            data: ecgValues
          }]
        }
      });
      
      // Simulate small location changes
      setLocation(prevLoc => ({
        lat: prevLoc.lat + (Math.random() * 0.002 - 0.001),
        lng: prevLoc.lng + (Math.random() * 0.002 - 0.001)
      }));
    }, 1000);
    
    return () => {
      clearInterval(interval);
      socket.close();
    };
  }, [patient]);

  if (!patient) {
    return <div className="loading">Loading patient data...</div>;
  }

  return (
    <div className="patient-detail-container">
      <div className="patient-detail-header">
        <Link to="/" className="back-button">← Back to Dashboard</Link>
        <h2>{patient.name}'s Health Monitor</h2>
        <div className={`status-badge ${patient.status}`}>{patient.status}</div>
      </div>
      
      <div className="patient-info-section">
        <div className="patient-profile">
          <h3>Patient Information</h3>
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Age:</span>
              <span className="value">{patient.age}</span>
            </div>
            <div className="info-item">
              <span className="label">Gender:</span>
              <span className="value">{patient.gender}</span>
            </div>
            <div className="info-item">
              <span className="label">Blood Type:</span>
              <span className="value">{patient.bloodType}</span>
            </div>
            <div className="info-item">
              <span className="label">Room:</span>
              <span className="value">{patient.room}</span>
            </div>
            <div className="info-item">
              <span className="label">Admission Date:</span>
              <span className="value">{patient.admissionDate}</span>
            </div>
            <div className="info-item">
              <span className="label">Attending Doctor:</span>
              <span className="value">{patient.doctor}</span>
            </div>
          </div>
          
          <div className="allergies">
            <h4>Allergies</h4>
            <ul>
              {patient.allergies.map((allergy, index) => (
                <li key={index}>{allergy}</li>
              ))}
            </ul>
          </div>
          
          <div className="diagnosis">
            <h4>Diagnosis</h4>
            <p>{patient.diagnosis}</p>
          </div>
        </div>
      </div>
      
      <div className="vitals-section">
        <h3>Real-time Vitals</h3>
        
        <div className="vitals-grid">
          <div className="vital-card heart-rate">
            <h4>Heart Rate</h4>
            <div className="vital-value">{vitals.heartRate} <span className="unit">BPM</span></div>
          </div>
          
          <div className="vital-card spo2">
            <h4>Oxygen Saturation</h4>
            <div className="vital-value">{vitals.spo2} <span className="unit">%</span></div>
          </div>
          
          <div className="vital-card temperature">
            <h4>Body Temperature</h4>
            <div className="vital-value">{vitals.temperature} <span className="unit">°C</span></div>
          </div>
        </div>
        
        <div className="ecg-container">
          <h4>ECG Monitor</h4>
          <div className="ecg-chart">
            <Line data={vitals.ecgData} options={{ responsive: true, animation: false }} />
          </div>
        </div>
        
        <div className="location-container">
          <h4>Patient Location</h4>
          <div className="map-widget" style={{ height: '300px' }}>
            <MapContainer center={location} zoom={18} style={{ height: '100%', width: '100%' }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              <Marker position={location}>
                <Popup>
                  {patient.name} is here <br /> Room: {patient.room}
                </Popup>
              </Marker>
            </MapContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientDetail;
