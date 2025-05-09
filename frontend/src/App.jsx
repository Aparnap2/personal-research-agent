import React, { useState, useEffect } from 'react';
import { ThemeProvider, CssBaseline, Container, Typography, Box, AppBar, Toolbar, CircularProgress, Alert, Paper, Grid, LinearProgress } from '@mui/material';
import AdbIcon from '@mui/icons-material/Adb';
import ResearchForm from './components/ResearchForm';
import ReportDisplay from './components/ReportDisplay';
import AgentLog from './components/AgentLog';
import { startResearch, getResearchStatus } from './services/api';
import theme from './theme';

function App() {
  const [projectId, setProjectId] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState('');
  const [pollingIntervalId, setPollingIntervalId] = useState(null);
  const [agentMessages, setAgentMessages] = useState([]);
  const [progress, setProgress] = useState(0);

  const [currentNodeMessage, setCurrentNodeMessage] = useState('Initializing...');

  const clearState = () => {
    setProjectId(null);
    setReportData(null);
    setIsStarting(false);
    setIsPolling(false);
    setError('');
    setAgentMessages([]);
    setProgress(0);
    if (pollingIntervalId) {
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }
  };

  const handleStartResearch = async (query) => {
    clearState();
    setIsStarting(true);
    setAgentMessages(prev => [...prev, { 
      timestamp: Date.now(), 
      text: 'Initializing research process...',
      type: 'system'
    }]);

    try {
      const response = await startResearch(query);
      if (!response?.project_id) {
        throw new Error('Failed to get project ID from server');
      }
      
      setProjectId(response.project_id);
      setIsStarting(false);
      setIsPolling(true);
      setAgentMessages(prev => [...prev, {
        timestamp: Date.now(),
        text: `Research started with ID: ${response.project_id}`,
        type: 'system'
      }]);
      
      // Start polling with proper error handling
      const pollForUpdates = async () => {
        try {
          const statusResponse = await getResearchStatus(response.project_id);
          
          // Safely handle messages
          const newMessages = Array.isArray(statusResponse?.messages) 
            ? statusResponse.messages.map(msg => ({
                timestamp: Date.now(),
                text: msg,
                type: 'agent'
              }))
            : [{ 
                timestamp: Date.now(), 
                text: 'Waiting for updates...',
                type: 'system'
              }];
          
          setAgentMessages(prev => [...prev, ...newMessages]);
          
          // Update progress if available from backend
          const calculatedProgress = statusResponse?.progress 
            ? statusResponse.progress 
            : Math.min(95, progress + 5); // Fallback incremental progress
          setProgress(calculatedProgress);

          if (statusResponse?.status === 'completed') {
            setReportData(statusResponse);
            setIsPolling(false);
            setProgress(100);
            clearInterval(intervalId);
            setAgentMessages(prev => [...prev, {
              timestamp: Date.now(),
              text: statusResponse.error_message 
                ? `Research completed with issues: ${statusResponse.error_message}`
                : 'Research completed successfully!',
              type: statusResponse.error_message ? 'error' : 'success'
            }]);
          }
        } catch (pollError) {
          console.error("Polling error:", pollError);
          setError('Failed to get research updates. Please check your connection.');
          setIsPolling(false);
          setProgress(0);
          clearInterval(intervalId);
          setAgentMessages(prev => [...prev, {
            timestamp: Date.now(),
            text: `Polling failed: ${pollError.message}`,
            type: 'error'
          }]);
        }
      };

      const intervalId = setInterval(async () => {
        try {
          const statusResponse = await getResearchStatus(response.project_id);
          // Ensure messages are always an array and map to objects with text and timestamp
          const newMessages = (statusResponse.messages || []).map(msg => 
            typeof msg === 'string' ? { timestamp: Date.now(), text: msg } : msg
          );
          setAgentMessages(newMessages);
          setCurrentNodeMessage(statusResponse.current_node_message || 'Processing...');
          
          // More sophisticated progress (example, needs backend support or better heuristics)
          // Example: if current_node_message contains "Phase X", update progress
          let newProgress = 0;
          if (statusResponse.current_node_message) {
            if (statusResponse.current_node_message.includes("Phase 1")) newProgress = 15;
            else if (statusResponse.current_node_message.includes("Phase 2") || statusResponse.current_node_message.includes("Web Search")) newProgress = 30;
            else if (statusResponse.current_node_message.includes("Phase 3") || statusResponse.current_node_message.includes("Synthesis")) newProgress = 50;
            else if (statusResponse.current_node_message.includes("Phase 4") || statusResponse.current_node_message.includes("Quantitative")) newProgress = 65;
            else if (statusResponse.current_node_message.includes("Phase 5") || statusResponse.current_node_message.includes("Statistical")) newProgress = 75;
            else if (statusResponse.current_node_message.includes("Phase 6") || statusResponse.current_node_message.includes("Visualization")) newProgress = 85;
            else if (statusResponse.current_node_message.includes("Phase 7") || statusResponse.current_node_message.includes("Report Compilation")) newProgress = 95;
          }
          if (statusResponse.completed) newProgress = 100;
          setProgress(newProgress);
    
          if (statusResponse.completed) {
            // ... (rest of completion logic)
          }
        } catch (pollError) {
          // ... (error handling)
        }
      }, 3000); // Poll every 3 seconds
      // ...
      setPollingIntervalId(intervalId);    

      return () => clearInterval(intervalId);    

    } catch (err) {
      console.error("Start research error:", err);
      setError(err.message || 'Failed to start research. Is the backend server running?');
      setIsStarting(false);
      setAgentMessages(prev => [...prev, {
        timestamp: Date.now(),
        text: `Initialization failed: ${err.message}`,
        type: 'error'
      }]);
    }
  };

  useEffect(() => {
    return () => {
      if (pollingIntervalId) clearInterval(pollingIntervalId);
    };
  }, [pollingIntervalId]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <AdbIcon sx={{ display: { xs: 'none', md: 'flex' }, mr: 1 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            AI Research Agent
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Paper elevation={2}>
              <Box sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>New Research Task</Typography>
                <ResearchForm onSubmit={handleStartResearch} isLoading={isStarting || isPolling} />
              </Box>
            </Paper>
            {(isStarting || isPolling) && (
          <Paper elevation={2} sx={{ mt: 2, p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="subtitle1" gutterBottom color="primary">
              Current Step: {currentNodeMessage}
            </Typography>
            <LinearProgress variant={isPolling && progress > 0 ? "determinate" : "indeterminate"} value={progress} sx={{mb: 1}} />
            <AgentLog messages={agentMessages} />
          </Paper>
        )}
          </Grid>
          <Grid item xs={12} md={8}>
            {reportData ? (
              <ReportDisplay reportData={reportData} />
            ) : (
              <Paper elevation={1} sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  {isStarting || isPolling ? 'Research in Progress' : 'Welcome!'}
                </Typography>
                <Typography>
                  {isStarting || isPolling 
                    ? 'Your results will appear here when ready...'
                    : 'Enter your research query to begin'}
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      </Container>
    </ThemeProvider>
  );
}

export default App;