import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDatasets } from '../context/DatasetContext';
import { toast } from 'react-toastify';
import { Upload, Database, AlertTriangle, TrendingUp, Activity, Calendar, X } from 'lucide-react';
import '../styles/Dashboard.css';

const DashboardPage = () => {
  const [datasetsSummary, setDatasetsSummary] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const { datasets, setCurrentDataset, removeDataset } = useDatasets();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await fetch('/api/datasets/dashboard-summary');
        if (!response.ok) throw new Error("Could not fetch dashboard data.");
        const data = await response.json();
        setDatasetsSummary(data);
      } catch (error) {
        toast.error(error.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchSummary();
  }, [datasets]); 

  const handleCardClick = (datasetSummary) => {
    const datasetToSelect = datasets.find(d => d.name === datasetSummary.filename);
    if (datasetToSelect) {
      setCurrentDataset(datasetToSelect);
      navigate('/data-table');
    } else {
      toast.error("Could not find the selected dataset.");
    }
  };

  const handleDeleteDataset = async (e, datasetName) => {
    e.stopPropagation();

    if (!window.confirm(`Are you sure you want to permanently delete '${datasetName}'? This action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/dataset/${datasetName}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Failed to delete dataset.");
        }

        setDatasetsSummary(prev => prev.filter(d => d.filename !== datasetName));
        removeDataset(datasetName);
        toast.success(`Dataset '${datasetName}' was successfully deleted.`);

    } catch (error) {
        toast.error(error.message);
    }
  };

  const activeDatasets = datasetsSummary.length;
  const qualityAlerts = datasetsSummary.filter(d => d.qualityScore < 70).length;
  const avgQuality = activeDatasets > 0 ? Math.round(datasetsSummary.reduce((sum, d) => sum + d.qualityScore, 0) / activeDatasets) : 0;
  const processingJobs = datasetsSummary.filter(d => d.status === "CLEANING").length;

  const getStatusColor = (status) => {
    if (status === "RAW") return "bg-red-100 text-red-800 border-red-200";
    if (status === "CLEANING") return "bg-yellow-100 text-yellow-800 border-yellow-200";
    return "bg-green-100 text-green-800 border-green-200";
  };

  const getStatusColorClass = (status) => {
    if (status === "CLEANED") return 'status-cleaned';
    if (status === "CLEANING") return 'status-cleaning';
    if (status === "RAW") return 'status-raw';
    return '';
  };
  
  const getQualityColorClass = (score) => {
    if (score >= 90) return 'high-quality';
    if (score >= 60) return 'medium-quality';
    return 'low-quality';
  };

  if (isLoading) {
    return <div>Loading Dashboard...</div>;
  }

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-left">
          <div className="metric-item">
            <Database className="metric-icon text-blue-600" />
            <div>
              <div className="metric-value">{activeDatasets} / 10</div>
              <div className="metric-label">ACTIVE DATASETS</div>
            </div>
          </div>
          <div className="metric-item">
            <AlertTriangle className="metric-icon text-red-600" />
            <div>
              <div className="metric-value">{qualityAlerts}</div>
              <div className="metric-label">QUALITY ALERTS</div>
            </div>
          </div>
          <div className="metric-item">
            <TrendingUp className="metric-icon text-green-600" />
            <div>
              <div className="metric-value">{avgQuality}%</div>
              <div className="metric-label">AVG. DATA QUALITY</div>
            </div>
          </div>
          <div className="metric-item">
            <Activity className="metric-icon text-orange-600" />
            <div>
              <div className="metric-value">{processingJobs}</div>
              <div className="metric-label">JOBS PROCESSING</div>
            </div>
          </div>
        </div>
        <div className="header-right">
          <button onClick={() => navigate('/upload')} className="upload-button">
            <Upload className="button-icon" />
            Upload Dataset
          </button>
          <div className="user-avatar">SB</div>
        </div>
      </header>

      <main className="dashboard-grid">
        {datasetsSummary.map((dataset) => {
          const qualityClass = getQualityColorClass(dataset.qualityScore);
          return (
            <div key={dataset.id} className="dataset-card" onClick={() => handleCardClick(dataset)} style={{ position: 'relative' }}>
              <button
                className="delete-dataset-btn"
                title={`Delete ${dataset.filename}`}
                onClick={(e) => handleDeleteDataset(e, dataset.filename)}
              >
                <X size={16} />
              </button>

              <div className="card-header">
                <div>
                  <h3 className="card-title">{dataset.filename}</h3>
                  <p className="card-subtitle">{dataset.size} | {dataset.rows.toLocaleString()} rows × {dataset.columns} cols</p>
                </div>
                <span className={`status-badge ${getStatusColorClass(dataset.status)}`}>{dataset.status}</span>
              </div>

              <div className="card-content">
                <div className="quality-score-section">
                  <span>DATA QUALITY SCORE</span>
                  {/* 2. Apply the dynamic color class to the percentage text */}
                  <span className={`quality-score ${qualityClass}`}>{dataset.qualityScore}%</span>
                </div>
                <div className="progress-bar-container">
                  {/* 3. Apply the dynamic color class to the progress bar */}
                  <div className={`progress-bar ${qualityClass}`} style={{ width: `${dataset.qualityScore}%` }}></div>
                </div>
                <div className="quality-breakdown-grid">
                  {/* 4. Add a title attribute for the tooltip */}
                  <div title={`Missing: ${dataset.missing}%`}>
                    <span className="breakdown-label">Missing</span>
                    <div className="progress-bar-container-small">
                      <div className="progress-bar-small" style={{ width: `${dataset.missing}%` }}></div>
                    </div>
                    <span className="breakdown-value">{dataset.missing}%</span>
                  </div>
                  <div title={`Duplicates: ${dataset.duplicates}%`}>
                    <span className="breakdown-label">Duplicates</span>
                    <div className="progress-bar-container-small">
                      <div className="progress-bar-small" style={{ width: `${dataset.duplicates}%` }}></div>
                    </div>
                    <span className="breakdown-value">{dataset.duplicates}%</span>
                  </div>
                  <div title={`Inconsistencies: ${dataset.inconsistencies}%`}>
                    <span className="breakdown-label">Inconsistencies</span>
                    <div className="progress-bar-container-small">
                      <div className="progress-bar-small" style={{ width: `${dataset.inconsistencies}%` }}></div>
                    </div>
                    <span className="breakdown-value">{dataset.inconsistencies}%</span>
                  </div>
                </div>
              </div>
              <div className="card-footer">
                <Calendar className="footer-icon" />
                <span>Uploaded: {dataset.lastModified}</span>
              </div>
            </div>
          );
        })}
      </main>
    </div>
  );
};

export default DashboardPage;