import React, { useState, useEffect } from 'react';
import { Route, Routes, Link } from 'react-router-dom';
import PatientDetail from './PatientDetail';
import './Dashboard.css';

const Dashboard = () => {
  const [darkMode, setDarkMode] = useState(localStorage.getItem('theme') === 'dark');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [patients, setPatients] = useState([]);
  const [doctor, setDoctor] = useState(null);
  const hospitalName = "MediCare General Hospital";

  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        // Updated endpoint to match your backend routes
        const response = await fetch('http://localhost:8000/');
        if (!response.ok) {
          throw new Error('Failed to fetch patients');
        }
        const data = await response.json();
        setPatients(data);
      } catch (error) {
        console.error("Error fetching patients:", error);
      }
    };

    const fetchDoctor = async () => {
      try {
        // Updated endpoint to match your backend routes
        const response = await fetch('http://localhost:8000');
        if (!response.ok) {
          throw new Error('Failed to fetch doctor info');
        }
        const data = await response.json();
        setDoctor(data);
      } catch (error) {
        console.error("Error fetching doctor info:", error);
      }
    };
    
    fetchPatients();
    fetchDoctor();
  }, []);

  const getStatusColor = (status) => {
    switch(status) {
      case 'critical': return 'status-critical';
      case 'needs attention': return 'status-attention';
      case 'normal': return 'status-normal';
      default: return '';
    }
  };

  const formatDate = (date) => {
    return date.toLocaleDateString('en-US', {
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  const HomePage = () => (
    <>
      <div className="dashboard-header">
        <h2>Patient Dashboard</h2>
      </div>
      
      <div className="patients-grid">
        {patients.length > 0 ? (
          patients.map(patient => (
            <Link to={`/patient/${patient.patient_id}`} key={patient._id}>
              <div className={`patient-card ${getStatusColor(patient.status)}`}>
                <h3>{patient.name}</h3>
                <p>ID: {patient.unique_id}</p>
                <p className="status">Status: {patient.status || 'N/A'}</p>
              </div>
            </Link>
          ))
        ) : (
          <p>No patients found</p>
        )}
      </div>
    </>
  );

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="hospital-info">
          <h1>{hospitalName}</h1>
        </div>
        
        {doctor && (
          <div className="doctor-profile">
            {doctor.avatar && <img src={doctor.avatar} alt="Doctor" className="doctor-avatar" />}
            <div className="doctor-details">
              <h2>{doctor.name || "Doctor Name"}</h2>
              <p>{doctor.email_id || 'Email not specified'}</p>
              <p>ID: {doctor.doc_id}</p>
            </div>
          </div>
        )}
        
        <div className="clock-widget">
          <div className="time">{currentTime.toLocaleTimeString()}</div>
          <div className="date">{formatDate(currentTime)}</div>
        </div>
        
        <div className="theme-toggle">
          <button onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          </button>
        </div>
      </aside>
      
      <main className="main-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/:id" element={<PatientDetail />} />
        </Routes>
      </main>
    </div>
  );
};

export default Dashboard;
