import React, { useState, useEffect, useMemo } from 'react';
import { Link, Routes, Route, useNavigate } from 'react-router-dom';
import PatientDetail from './PatientDetail';
import AddPatientPage from './AddPatientPage';
import './Dashboard.css';

const Dashboard = () => {
  const [darkMode, setDarkMode] = useState(localStorage.getItem('theme') === 'dark');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [patients, setPatients] = useState([]);
  const [doctor, setDoctor] = useState(null);
  const hospitalName = "MediCare General Hospital";

  // Dark mode effect
  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // Clock effect - separated from other state updates
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Data fetching effect
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/');
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }

        const patientsData = await response.json();
        console.log('Fetched patients:', patientsData); // Debug log
        setPatients(patientsData);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };
    
    fetchData();
  }, []);

  // Memoize functions and values that don't need to change on every render
  const getStatusColor = useMemo(() => (status) => {
    switch(status) {
      case 'critical': return 'status-critical';
      case 'needs attention': return 'status-attention';
      case 'normal': return 'status-normal';
      default: return '';
    }
  }, []);

  const formatDate = useMemo(() => (date) => {
    return date.toLocaleDateString('en-US', {
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  }, []);

  const HomePage = () => {
    const navigate = useNavigate();
    
    return (
      <>
        <div className="dashboard-header">
          <div className="header-content">
            <h2>Patient Dashboard</h2>
            <button 
              className="add-patient-btn"
              onClick={() => navigate('/add-patient')}
            >
              Add New Patient
            </button>
          </div>
        </div>
        
        <div className="patients-grid">
          {patients.length > 0 ? (
            patients.map(patient => (
              <Link to={`/patient/${patient.unique_id}`} key={patient._id}>
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
  };

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
          <Route path="/add-patient" element={<AddPatientPage />} />
          <Route path="/patient/:id" element={<PatientDetail />} />
        </Routes>
      </main>
    </div>
  );
};

export default Dashboard;
