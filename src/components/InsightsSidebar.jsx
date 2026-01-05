import React from 'react';
import '../styles/InsightsSidebar.css';
import { FiChevronRight, FiChevronLeft } from 'react-icons/fi';

const Loader = () => (
  <div className="loader-container">
    <div className="loader"></div>
    <p>Analyzing...</p>
  </div>
);

const AnalysisResult = ({ analysis }) => {
  if (!analysis) return null;

  if (analysis.status && analysis.status !== 'SUCCESS') {
    return (
      <div className="error-message">
        <strong>{analysis.status}</strong>
        <p>{analysis.message || analysis.error}</p>
      </div>
    );
  }
  
  const aiResponse = analysis.result;
  if (!aiResponse) return <div className="error-message"><p>No AI response data available.</p></div>;

  return (
    <div className="analysis-report">
      {/* --- AI Recommendation Section --- */}
      <div className="recommendation-box">
        <div className="recommendation-label">AI Recommendation</div>
        <div className="recommendation-text">{aiResponse.recommendation}</div>
      </div>

      {/* --- Explanation Section --- */}
      <details className="explanation-dropdown">
        <summary>Explain Why...</summary>
        <p className="explanation-text">{aiResponse.reasoning_summary}</p>
      </details>

      {/* --- Assumptions Section --- */}
      {aiResponse.assumptions && (
        <div className="assumptions-box">
          <div className="box-label">Assumptions Made</div>
          <ul>
            {aiResponse.assumptions.map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </div>
      )}

      {/* --- Warning Section --- */}
      {aiResponse.warning && (
        <div className="warning-box">
          <div className="box-label">⚠️ Warning</div>
          <p>{aiResponse.warning}</p>
        </div>
      )}
    </div>
  );
};

const InsightsSidebar = ({ column, aiAnalysis, isLoading, sidebarState, setSidebarState }) => {
  if (!column || sidebarState === 'closed') return null;

  const handleClose = () => setSidebarState('closed');
  const handleToggle = () => setSidebarState(sidebarState === 'open' ? 'minimized' : 'open');

  const sidebarClasses = `insights-sidebar ${sidebarState === 'minimized' ? 'minimized' : ''}`;

  return (
    <div className={sidebarClasses}>
      <div className="sidebar-toggle-btn" onClick={handleToggle}>
        {sidebarState === 'minimized' ? <FiChevronLeft /> : <FiChevronRight />}
      </div>

      <div className="sidebar-header">
        <h3>Insights for "{column.getColDef().headerName}"</h3>
        <button onClick={handleClose} className="close-btn">&times;</button>
      </div>
      <div className="sidebar-content">
        {isLoading ? (
          <Loader />
        ) : (
          <AnalysisResult analysis={aiAnalysis} />
        )}
      </div>
    </div>
  );
};

export default InsightsSidebar;