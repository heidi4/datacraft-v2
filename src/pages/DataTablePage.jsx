import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import DataTable from '../components/DataTable';
import Papa from 'papaparse';
import { useDatasets } from '../context/DatasetContext';
import { toast } from 'react-toastify';
import { FaFileCsv, FaChartBar } from "react-icons/fa";
import { FiChevronDown, FiUpload, FiTool, FiCpu } from "react-icons/fi";
import InsightsSidebar from '../components/InsightsSidebar';
import StatisticsSidebar from '../components/StatisticsSidebar';
import '../styles/DataTablePage.css';

const ThemeSelector = ({ theme, setTheme }) => (
    <select className="theme-selector" value={theme} onChange={e => setTheme(e.target.value)}>
        <option value="ag-theme-alpine">Alpine</option>
        <option value="ag-theme-balham">Balham</option>
        <option value="ag-theme-quartz">Quartz</option>
        <option value="ag-theme-material">Material</option>
    </select>
);

const QuickActions = ({ onAction }) => {
    const [isOpen, setIsOpen] = useState(false);
    const ref = useRef(null);
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (ref.current && !ref.current.contains(event.target)) setIsOpen(false);
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [ref]);
    const handleSelect = (action) => {
        onAction(action);
        setIsOpen(false);
    };
    return (
        <div className="quick-actions-menu" ref={ref}>
            <button className="quick-actions-button" onClick={() => setIsOpen(o => !o)}>
                <FiTool size={16} /> Quick Actions <FiChevronDown size={16} style={{ transform: isOpen ? 'rotate(180deg)' : 'none' }}/>
            </button>
            {isOpen && (
                <div className="quick-actions-dropdown">
                    <button onClick={() => handleSelect('drop_na_rows')}><strong>Drop Rows with Missing Values</strong><span>Deletes any row with at least one empty cell.</span></button>
                    <button onClick={() => handleSelect('drop_duplicate_rows')}><strong>Drop Duplicate Rows</strong><span>Deletes all rows that are exact duplicates.</span></button>
                </div>
            )}
        </div>
    );
};

const DataTablePage = () => {
    const { datasets, currentDataset, setCurrentDataset } = useDatasets();
    const [rowData, setRowData] = useState([]);
    const [error, setError] = useState('');
    const [theme, setTheme] = useState('ag-theme-alpine');
    
    // --- Existing states for statistics
    const [datasetMetrics, setDatasetMetrics] = useState(null);
    const [areMetricsLoading, setAreMetricsLoading] = useState(true);

    // --- New states for Co-pilot Diagnostic Report
    const [diagnosticReport, setDiagnosticReport] = useState(null);
    const [isReportLoading, setIsReportLoading] = useState(false);

    // --- Sidebar management
    const [statsSidebarState, setStatsSidebarState] = useState('closed'); // 'open' or 'closed'
    const [sidebarMode, setSidebarMode] = useState('statistics'); // 'statistics' or 'copilot'
    
    // --- Co-pilot simulation controls
    const [goal, setGoal] = useState('stable_forecasting');
    const [targetVariable, setTargetVariable] = useState('');

    const metricsPollingRef = useRef(null);
    const diagnosticPollingRef = useRef(null);
    const insightsPollingRef = useRef(null);
    const [insightsSidebarState, setInsightsSidebarState] = useState('closed');
    const [sidebarColumn, setSidebarColumn] = useState(null);
    const [aiAnalysis, setAiAnalysis] = useState(null);
    const [isInsightsLoading, setIsInsightsLoading] = useState(false);

    const [treatmentPlans, setTreatmentPlans] = useState(null); 
    const [arePlansLoading, setArePlansLoading] = useState(false);
    const plansPollingRef = useRef(null);

    const [simulationResults, setSimulationResults] = useState(null);
    const [isSimulating, setIsSimulating] = useState(false);
    const simulationPollingRef = useRef(null);
    const [simulationWarnings, setSimulationWarnings] = useState([]);

    const [isApplyingPlan, setIsApplyingPlan] = useState(false);
    
    const loadData = useCallback(async () => {
        if (!currentDataset) return;
        try {
            const urlToFetch = `${currentDataset.path}?t=${Date.now()}`;
            const response = await fetch(urlToFetch);
            if (!response.ok) throw new Error('Failed to fetch dataset');
            const csvText = await response.text();
            
            Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (result) => {
                    if (result.data && result.data.length > 0) {
                        setRowData(result.data);
                        setError('');
                    } else {
                        setRowData([]);
                        setError('No data rows found in CSV');
                    }
                },
                error: () => { setError('Failed to parse CSV file.'); }
            });
        } catch (e) { setError(e.message || 'Could not load or process the file.'); }
    }, [currentDataset]);

    const fetchMetrics = useCallback(async () => {
        if (!currentDataset) return;
        if (metricsPollingRef.current) clearInterval(metricsPollingRef.current);
        setAreMetricsLoading(true);

        const fetchData = async () => {
            try {
                const response = await fetch(`/api/dataset/${currentDataset.name}/statistics`);
                if (response.status === 202) return;
                if (!response.ok) throw new Error('Failed to fetch statistics.');

                const data = await response.json();
                setDatasetMetrics(data);
                setAreMetricsLoading(false);
                clearInterval(metricsPollingRef.current);
            } catch (err) {
                toast.error(err.message);
                setAreMetricsLoading(false);
                clearInterval(metricsPollingRef.current);
            }
        };
        fetchData();
        metricsPollingRef.current = setInterval(fetchData, 3000);
    }, [currentDataset]);

    const fetchDiagnosticReport = useCallback(async () => {
        if (!currentDataset) return;
        if (diagnosticPollingRef.current) clearInterval(diagnosticPollingRef.current);
        setIsReportLoading(true);

        const fetchData = async () => {
            try {
                const response = await fetch(`/api/dataset/${currentDataset.name}/diagnostics`);
                if (response.status === 202) return;
                if (!response.ok) throw new Error('Failed to fetch diagnostic report.');
                const data = await response.json();
                setDiagnosticReport(data);
                setIsReportLoading(false);
                clearInterval(diagnosticPollingRef.current);
            } catch (err) {
                toast.error(err.message);
                setIsReportLoading(false);
                clearInterval(diagnosticPollingRef.current);
            }
        };
        fetchData();
        diagnosticPollingRef.current = setInterval(fetchData, 3000);
    }, [currentDataset]);

    const handleOpenStatistics = () => {
        setSidebarMode('statistics');
        setInsightsSidebarState('closed');
        setStatsSidebarState('open');
        // Ensure latest stats are loaded
        if (!datasetMetrics || datasetMetrics.filename !== currentDataset?.name) {
            fetchMetrics();
        }
    };

    const handleOpenAdvancedCoPilot = () => {
        setSidebarMode('copilot');
        setInsightsSidebarState('closed');
        setStatsSidebarState('open');
         // Ensure latest diagnostic report is loaded
        if (!diagnosticReport || diagnosticReport.filename !== currentDataset?.name) {
            fetchDiagnosticReport();
        }
    };

    const forceRefreshMetrics = useCallback(() => {
        if (!currentDataset) return;
        toast.info("Refreshing dataset statistics...");
        setDatasetMetrics(null);
        fetch(`/api/dataset/${currentDataset.name}/refresh-statistics`, { method: 'POST' })
            .then(res => {
                if(res.ok) setTimeout(fetchMetrics, 2000);
                else throw new Error("Failed to start statistics refresh job.");
            })
            .catch(err => toast.error(err.message));
    }, [currentDataset, fetchMetrics]);

    const handleActionComplete = useCallback(() => {
        loadData();
        forceRefreshMetrics();
    }, [loadData, forceRefreshMetrics]);

    const handleGeneratePlans = useCallback(async () => {
        if (!currentDataset || !targetVariable) {
            toast.warn("Please select a dataset and target variable first.");
            return;
        }

        if (plansPollingRef.current) clearInterval(plansPollingRef.current);
        setArePlansLoading(true);
        setTreatmentPlans(null); // Clear previous plans
        const toastId = toast.loading("Submitting request to AI Co-pilot...");

        try {
            const response = await fetch(`/api/dataset/${currentDataset.name}/generate-plans`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_variable: targetVariable,
                    goal: goal,
                }),
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Failed to start plan generation job.");
            }

            const { job_id } = await response.json();
            toast.update(toastId, { render: "Job submitted. Waiting for AI to generate plans...", type: 'info', isLoading: true });

            plansPollingRef.current = setInterval(async () => {
                const statusResponse = await fetch(`/api/analyze/status/${job_id}`);
                const data = await statusResponse.json();

                if (data.status !== 'PENDING') {
                    clearInterval(plansPollingRef.current);
                    setArePlansLoading(false);
                    
                    if (data.status === 'SUCCESS') {
                        setTreatmentPlans(data.result);
                        toast.update(toastId, { render: "AI Treatment Plans generated successfully!", type: 'success', isLoading: false, autoClose: 5000 });
                        // NOTE: In the next step, we will display these plans.
                        // For now, we can log them to see the result.
                        console.log("Generated Plans:", data.result);
                    } else {
                        setTreatmentPlans(null);
                        toast.update(toastId, { render: `Task failed: ${data.error || "An unknown error occurred."}`, type: 'error', isLoading: false, autoClose: 7000 });
                    }
                }
            }, 5000); // Poll every 5 seconds for this potentially longer task

        } catch (error) {
            setArePlansLoading(false);
            toast.update(toastId, { render: error.message, type: 'error', isLoading: false, autoClose: 7000 });
        }

    }, [currentDataset, targetVariable, goal]);

    const handleRunTask = useCallback(async (taskType, column) => {
        if (!currentDataset) {
            toast.error("Please select a dataset first.");
            return;
        }

        const modificationTasks = ['delete_column', 'impute_mean', 'impute_median', 'impute_mode', 'impute_constant'];
        let taskParams = {};

        if (taskType === 'impute_constant') {
            const customValue = window.prompt("Enter the value to fill missing cells with:", "");
            if (customValue === null) {
                return;
            }
            taskParams.value = customValue;
        }

        if (taskType === 'diagnosis') {
            if (insightsPollingRef.current) clearInterval(insightsPollingRef.current);
            setSidebarColumn(column);
            setAiAnalysis(null);
            setIsInsightsLoading(true);
            setInsightsSidebarState('open');
            setStatsSidebarState('closed');
        }

        const toastId = toast.loading(`Submitting task '${taskType}'...`);
        try {
            const response = await fetch('/api/submit_task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_name: currentDataset.name,
                    column_name: column.getColId(),
                    task_type: taskType,
                    task_params: taskParams
                }),
            });
            if (!response.ok) throw new Error('Failed to submit job.');
            
            const { job_id } = await response.json();
            toast.update(toastId, { render: `Job '${taskType}' submitted.`, type: 'info', isLoading: false, autoClose: 3000 });

            if (modificationTasks.includes(taskType)) {
                setTimeout(() => {
                    toast.success("Action complete! Refreshing data...");
                    loadData();
                }, 2000); 
                return;
            }

            insightsPollingRef.current = setInterval(async () => {
                const statusResponse = await fetch(`/api/analyze/status/${job_id}`);
                const data = await statusResponse.json();
                if (data.status !== 'PENDING') {
                    clearInterval(insightsPollingRef.current);
                    setIsInsightsLoading(false);
                    setAiAnalysis(data);
                    if (data.status !== 'SUCCESS') {
                        toast.error(`Task failed: ${data.error || "An unknown error occurred."}`);
                    } else {
                        toast.success("AI Analysis complete!");
                    }
                }
            }, 3000);
        } catch (error) {
            toast.update(toastId, { render: error.message, type: 'error', isLoading: false, autoClose: 5000 });
            if (taskType === 'diagnosis') {
                setAiAnalysis({ status: "FAILURE", error: error.message });
                setIsInsightsLoading(false);
            }
        }
    }, [currentDataset, loadData]);

    const handleApplyPlan = useCallback(async (plan) => {
        if (!currentDataset) return;
        
        if (!window.confirm(`Are you sure you want to apply "${plan.name}"? This will permanently modify the dataset.`)) {
            return;
        }

        setIsApplyingPlan(true);
        const toastId = toast.loading("Applying plan strategies to dataset...");

        try {
            // This matches the endpoint we added to main.py
            const response = await fetch(`/api/dataset/${currentDataset.name}/apply-plan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    python_code: plan.python_code, // The hidden field we added to the AI JSON
                    plan_name: plan.name
                }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Failed to apply plan.");
            }

            const result = await response.json();
            
            // Wait for the background task if it's async, or if it returns immediately:
            // (Our backend endpoint returns a job_id, so we technically should poll, 
            // but for simplicity, let's assume the user waits or we do a quick timeout 
            // since the endpoint you added earlier returns a job_id).
            
            // IF YOUR ENDPOINT RETURNS A JOB_ID (Async way):
            // We should poll similar to generate-plans. 
            // Let's do a quick polling implementation here for robustness.
            
            const jobId = result.job_id;
            const pollInterval = setInterval(async () => {
                const statusRes = await fetch(`/api/analyze/status/${jobId}`);
                const statusData = await statusRes.json();
                
                if (statusData.status !== 'PENDING') {
                    clearInterval(pollInterval);
                    setIsApplyingPlan(false);
                    
                    if (statusData.status === 'SUCCESS') {
                         toast.update(toastId, { render: "Plan applied successfully! Reloading data...", type: 'success', isLoading: false, autoClose: 3000 });
                         
                         // Refresh everything
                         handleActionComplete(); 
                         // Reset simulation state so user goes back to start or sees updated stats
                         setTreatmentPlans(null); 
                         setSimulationResults(null);
                    } else {
                        toast.update(toastId, { render: `Application failed: ${statusData.error}`, type: 'error', isLoading: false, autoClose: 5000 });
                    }
                }
            }, 2000);

        } catch (error) {
            setIsApplyingPlan(false);
            toast.update(toastId, { render: error.message, type: 'error', isLoading: false, autoClose: 5000 });
        }
    }, [currentDataset, handleActionComplete]);

    const handleQuickAction = useCallback(async (actionType) => {
        if (!currentDataset) {
            toast.warn("Please select a dataset first.");
            return;
        }

        const actionName = actionType === 'drop_na_rows' ? 'drop all rows with missing values' : 'drop all duplicate rows';
        if (!window.confirm(`This will permanently modify the '${currentDataset.name}' file by attempting to ${actionName}. Are you sure you want to continue?`)) {
            return;
        }

        const toastId = toast.loading(`Performing '${actionName}'...`);
        try {
            const response = await fetch('/api/dataset/clean', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_name: currentDataset.name,
                    action_type: actionType,
                }),
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Failed to start cleaning job.");
            }

            const result = await response.json();
            
            setTimeout(() => {
                toast.update(toastId, { render: "Cleaning successful! Table has been refreshed.", type: "success", isLoading: false, autoClose: 5000 });
                handleActionComplete();
            }, 2000);

        } catch (error) {
            toast.update(toastId, { render: error.message, type: "error", isLoading: false, autoClose: 5000 });
        }
    }, [currentDataset, handleActionComplete]);

    const handleRunSimulation = useCallback(async () => {
        if (!currentDataset || !treatmentPlans) return;

        if (simulationPollingRef.current) clearInterval(simulationPollingRef.current);
        setIsSimulating(true);
        setSimulationResults(null);
        const toastId = toast.loading("Starting impact simulation... This may take a few minutes.");

        try {
            const response = await fetch(`/api/dataset/${currentDataset.name}/run-simulation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    plans: treatmentPlans,
                    target_variable: targetVariable,
                    goal: goal === 'stable_forecasting' ? 'regression' : 'classification', // Example mapping
                }),
            });
            if (!response.ok) throw new Error((await response.json()).detail || 'Failed to start simulation.');

            const { job_id } = await response.json();
            toast.update(toastId, { render: "Simulation running. Validating plans against probe models...", isLoading: true });

            simulationPollingRef.current = setInterval(async () => {
                const statusResponse = await fetch(`/api/analyze/status/${job_id}`);
                const data = await statusResponse.json();

                if (data.status !== 'PENDING') {
                    clearInterval(simulationPollingRef.current);
                    setIsSimulating(false); // <--- THIS MUST BE HERE TO STOP SPINNER
                    
                    if (data.status === 'SUCCESS') {
                        setSimulationResults(data.result);
                        setSimulationWarnings(data.warnings || []);
                        toast.update(toastId, { render: "Simulation complete!", type: 'success', isLoading: false, autoClose: 5000 });
                    } else {
                        // It failed
                        toast.update(toastId, { render: `Simulation failed: ${data.error}`, type: 'error', isLoading: false, autoClose: 7000 });
                    }
                }
            }, 5000);

        } catch (error) {
            setIsSimulating(false);
            toast.update(toastId, { render: error.message, type: "error", isLoading: false, autoClose: 7000 });
        }
    }, [currentDataset, treatmentPlans, targetVariable, goal]);

    const columnDefs = useMemo(() => {
        if (!datasetMetrics?.columnStats) return [];
        return datasetMetrics.columnStats.map(stat => ({
            headerName: stat.column.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            field: stat.column,
            headerComponentParams: { description: stat.dataType },
            sortable: true, filter: true, editable: true, enableRowGroup: true,
            ...(datasetMetrics.columnStats.indexOf(stat) === 0 && { checkboxSelection: true, headerCheckboxSelection: true }),
        }));
    }, [datasetMetrics]);

    useEffect(() => {
        if (currentDataset) {
            loadData();
            fetchMetrics();
        }
        return () => {
            if (metricsPollingRef.current) clearInterval(metricsPollingRef.current);
            if (insightsPollingRef.current) clearInterval(insightsPollingRef.current);
        };
    }, [currentDataset, loadData, fetchMetrics]);

    useEffect(() => {
        if (!currentDataset && datasets.length > 0) setCurrentDataset(datasets[0]);
    }, [datasets, currentDataset, setCurrentDataset]);

    const handleDatasetChange = (e) => {
        const selectedFile = datasets.find(d => d.name === e.target.value);
        setCurrentDataset(selectedFile);
    };

    return (
        <div className="page-container" style={{ display: 'flex' }}>
            <div className={`datatable-page-main-content ${statsSidebarState === 'open' ? 'stats-sidebar-open' : ''} ${insightsSidebarState === 'open' ? 'insights-sidebar-open' : ''}`}>
                <header className="page-header animated-component">
                    <div><h1>Data Table</h1><p>Explore and analyze your datasets</p></div>
                    <div className="header-actions">
                        <ThemeSelector theme={theme} setTheme={setTheme} />
                        {datasets.length > 0 && currentDataset ? (
                            <select className="file-selector" value={currentDataset.name} onChange={handleDatasetChange}>
                                {datasets.map(dataset => (<option key={dataset.name} value={dataset.name}>{dataset.name}</option>))}
                            </select>
                        ) : (<p>No datasets available.</p>)}
                    </div>
                </header>
                <div className="dataset-info-bar animated-component" style={{ animationDelay: '0.2s' }}>
                    <div className="file-details">
                        <FaFileCsv />
                        <div>
                            <strong>{currentDataset?.name || 'No file selected'}</strong>
                            <span>{rowData.length} rows &times; {columnDefs.length > 0 ? columnDefs.length : '...'} columns</span>
                        </div>
                    </div>
                    <div className="header-actions">
                        <QuickActions onAction={handleQuickAction} />
                        <button className={`statistics-button ${statsSidebarState === 'open' && sidebarMode === 'statistics' ? 'active' : ''}`} onClick={handleOpenStatistics}>
                            <FaChartBar /> Statistics
                        </button>
                        <button className={`co-pilot-button ${statsSidebarState === 'open' && sidebarMode === 'copilot' ? 'active' : ''}`} onClick={handleOpenAdvancedCoPilot}>
                            <FiCpu size={16}/> Co-pilot
                        </button>
                        <button className="export-button"><FiUpload /> Export</button>
                    </div>
                </div>
                {error ? (<div className="error-message">{error}</div>) : (
                    <div className='animated-component'>
                        <DataTable
                            rowData={rowData}
                            columnDefs={columnDefs}
                            theme={theme}
                            onRunTask={handleRunTask}
                            onRefresh={loadData}
                        />
                    </div>
                )}
            </div>
            <StatisticsSidebar
                statistics={datasetMetrics}
                isLoading={areMetricsLoading}
                sidebarState={statsSidebarState}
                setSidebarState={setStatsSidebarState}
                sidebarMode={sidebarMode}
                diagnosticReport={diagnosticReport}
                isReportLoading={isReportLoading}
                goal={goal}
                setGoal={setGoal}
                targetVariable={targetVariable}
                setTargetVariable={setTargetVariable}
                onGeneratePlans={handleGeneratePlans}
                // --- Pass down new state and placeholder function ---
                arePlansLoading={arePlansLoading}
                treatmentPlans={treatmentPlans}
                onRunSimulation={handleRunSimulation}
                isSimulating={isSimulating}
                simulationResults={simulationResults}
                simulationWarnings={simulationWarnings}
                onApplyPlan={handleApplyPlan}
                isApplying={isApplyingPlan}
            />
            <InsightsSidebar
                column={sidebarColumn}
                aiAnalysis={aiAnalysis}
                isLoading={isInsightsLoading}
                sidebarState={insightsSidebarState}
                setSidebarState={setInsightsSidebarState}
            />
        </div>
    );
};
export default DataTablePage;