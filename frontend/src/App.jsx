import React, { useState, useEffect } from 'react';
import { 
  ThemeProvider, 
  CssBaseline, 
  Container, 
  Typography, 
  Box, 
  AppBar, 
  Toolbar, 
  CircularProgress, 
  Alert, 
  Paper, 
  Grid, 
  LinearProgress,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
  useMediaQuery,
  Tooltip,
  Fade,
  Chip
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import HistoryIcon from '@mui/icons-material/History';
import DashboardIcon from '@mui/icons-material/Dashboard';
import SearchIcon from '@mui/icons-material/Search';
import SettingsIcon from '@mui/icons-material/Settings';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import GitHubIcon from '@mui/icons-material/GitHub';
import ResearchForm from './components/ResearchForm';
import ReportDisplay from './components/ReportDisplay';
import AgentLog from './components/AgentLog';
import ProjectHistory from './components/ProjectHistory';
import Dashboard from './components/DashboardWithRealData';
import Settings from './components/Settings';
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
  const [currentNodeMessage, setCurrentNodeMessage] = useState('Awaiting research task...');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeView, setActiveView] = useState('research'); // 'research', 'history', 'dashboard', 'settings'
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const clearState = () => {
    setProjectId(null);
    setReportData(null);
    setIsStarting(false);
    setIsPolling(false);
    setError('');
    setAgentMessages([]);
    setProgress(0);
    setCurrentNodeMessage('Awaiting research task...');
    if (pollingIntervalId) {
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }
  };
  
  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };
  
  const handleViewChange = (view) => {
    setActiveView(view);
    if (isMobile) {
      setDrawerOpen(false);
    }
  };
  
  const handleSelectHistoryProject = (selectedProjectId) => {
    // In a real implementation, this would fetch the project data
    setProjectId(selectedProjectId);
    setActiveView('research');
    // Mock loading a report from history
    setIsPolling(true);
    setTimeout(() => {
      // This is just a mock - in reality you'd fetch the actual report data
      const mockHistoricalReport = {
        project_id: selectedProjectId,
        completed: true,
        report_markdown: "# Historical Report\n\nThis is a previously generated report that was loaded from history.",
        charts: [],
        current_node_message: "Completed"
      };
      setReportData(mockHistoricalReport);
      setIsPolling(false);
    }, 1500);
  };

  const handleStartResearch = async (query) => {
    clearState();
    setIsStarting(true);
    setAgentMessages([{ 
      timestamp: new Date().toISOString(), 
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
        timestamp: new Date().toISOString(),
        text: `Research started with ID: ${response.project_id}`,
        type: 'success'
      }]);
    } catch (err) {
      console.error("Start research error:", err);
      setError(err.message || 'Failed to start research. Is the backend server running?');
      setIsStarting(false);
      setAgentMessages(prev => [...prev, {
        timestamp: new Date().toISOString(),
        text: `Initialization failed: ${err.message}`,
        type: 'error'
      }]);
    }
  };

  useEffect(() => {
    if (isPolling && projectId) {
      const intervalId = setInterval(async () => {
        try {
          const statusResponse = await getResearchStatus(projectId);
          
          // Update messages
          const newMessages = (statusResponse.messages || []).map(msg => ({
            timestamp: msg.timestamp || new Date().toISOString(),
            text: msg.text || 'Unknown message',
            type: msg.type || 'info'
          }));
          setAgentMessages(newMessages);

          // Update current node message
          setCurrentNodeMessage(statusResponse.current_node_message || 'Processing...');

          // Update progress based on phase
          let newProgress = 0;
          const nodeMsg = statusResponse.current_node_message?.toLowerCase() || '';
          if (nodeMsg.includes('phase 1') || nodeMsg.includes('planning')) newProgress = 15;
          else if (nodeMsg.includes('phase 2') || nodeMsg.includes('search') || nodeMsg.includes('scraping')) newProgress = 30;
          else if (nodeMsg.includes('phase 3') || nodeMsg.includes('synthesis')) newProgress = 50;
          else if (nodeMsg.includes('phase 4') || nodeMsg.includes('quantitative')) newProgress = 65;
          else if (nodeMsg.includes('phase 5') || nodeMsg.includes('statistical')) newProgress = 75;
          else if (nodeMsg.includes('phase 6') || nodeMsg.includes('visualization')) newProgress = 85;
          else if (nodeMsg.includes('phase 7') || nodeMsg.includes('report')) newProgress = 95;
          if (statusResponse.completed) newProgress = 100;
          setProgress(newProgress);

          // Stop polling if completed
          if (statusResponse.completed) {
            setReportData(statusResponse);
            setIsPolling(false);
            setProgress(100);
            clearInterval(intervalId);
            setPollingIntervalId(null);
            setAgentMessages(prev => [...prev, {
              timestamp: new Date().toISOString(),
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
          setPollingIntervalId(null);
          setAgentMessages(prev => [...prev, {
            timestamp: new Date().toISOString(),
            text: `Polling failed: ${pollError.message}`,
            type: 'error'
          }]);
        }
      }, 3000); // Poll every 3 seconds
      setPollingIntervalId(intervalId);
      return () => clearInterval(intervalId);
    }
  }, [isPolling, projectId]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        {/* App Bar */}
        <AppBar position="static" sx={{ bgcolor: 'primary.dark', zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={toggleDrawer}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <AnalyticsIcon sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
              Research Intelligence Platform
            </Typography>
            <Tooltip title="View on GitHub">
              <IconButton color="inherit" component="a" href="https://github.com/Aparnap2/personal-research-agent" target="_blank">
                <GitHubIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Help">
              <IconButton color="inherit">
                <HelpOutlineIcon />
              </IconButton>
            </Tooltip>
          </Toolbar>
          {(isStarting || isPolling) && (
            <LinearProgress 
              variant={progress > 0 ? "determinate" : "indeterminate"} 
              value={progress} 
              sx={{ height: 4 }} 
            />
          )}
        </AppBar>
        
        {/* Navigation Drawer */}
        <Drawer
          variant={isMobile ? "temporary" : "persistent"}
          open={isMobile ? drawerOpen : true}
          onClose={toggleDrawer}
          sx={{
            width: 240,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 240,
              boxSizing: 'border-box',
              top: ['48px', '56px', '64px'],
              height: 'auto',
              bottom: 0,
            },
          }}
        >
          <List>
            <ListItem 
              button 
              selected={activeView === 'dashboard'} 
              onClick={() => handleViewChange('dashboard')}
            >
              <ListItemIcon>
                <DashboardIcon color={activeView === 'dashboard' ? 'primary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText primary="Dashboard" />
            </ListItem>
            <ListItem 
              button 
              selected={activeView === 'research'} 
              onClick={() => handleViewChange('research')}
            >
              <ListItemIcon>
                <SearchIcon color={activeView === 'research' ? 'primary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText primary="Research" />
            </ListItem>
            <ListItem 
              button 
              selected={activeView === 'history'} 
              onClick={() => handleViewChange('history')}
            >
              <ListItemIcon>
                <HistoryIcon color={activeView === 'history' ? 'primary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText primary="History" />
            </ListItem>
            <Divider />
            <ListItem 
              button 
              selected={activeView === 'settings'} 
              onClick={() => handleViewChange('settings')}
            >
              <ListItemIcon>
                <SettingsIcon color={activeView === 'settings' ? 'primary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText primary="Settings" />
            </ListItem>
          </List>
        </Drawer>
        
        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            ml: isMobile ? 0 : '240px',
            transition: theme.transitions.create('margin', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
          }}
        >
          <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
            {error && (
              <Alert 
                severity="error" 
                sx={{ mb: 3, borderRadius: 2 }}
                action={
                  <IconButton
                    aria-label="close"
                    color="inherit"
                    size="small"
                    onClick={() => setError('')}
                  >
                    <CloseIcon fontSize="inherit" />
                  </IconButton>
                }
              >
                {error}
              </Alert>
            )}
            
            {/* Dashboard View */}
            {activeView === 'dashboard' && (
              <Dashboard />
            )}
            
            {/* Research View */}
            {activeView === 'research' && (
              <Grid container spacing={3}>
                {/* Left Sidebar: Form and Log */}
                <Grid item xs={12} md={4}>
                  <Paper elevation={3} sx={{ p: 3, mb: 2, borderRadius: 2 }}>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main', display: 'flex', alignItems: 'center' }}>
                      <SearchIcon sx={{ mr: 1 }} />
                      New Research Task
                    </Typography>
                    <ResearchForm onSubmit={handleStartResearch} isLoading={isStarting || isPolling} />
                  </Paper>
                  {(isStarting || isPolling) && (
                    <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 'medium', color: 'primary.main' }}>
                          Current Step: 
                        </Typography>
                        <Chip 
                          label={currentNodeMessage} 
                          color="primary" 
                          variant="outlined"
                          size="small"
                        />
                      </Box>
                      <LinearProgress 
                        variant={progress > 0 ? "determinate" : "indeterminate"} 
                        value={progress} 
                        sx={{ mb: 2, height: 8, borderRadius: 4 }} 
                      />
                      <AgentLog messages={agentMessages} />
                    </Paper>
                  )}
                </Grid>
                {/* Main Content: Report */}
                <Grid item xs={12} md={8}>
                  {reportData ? (
                    <ReportDisplay reportData={reportData} />
                  ) : (
                    <Paper elevation={3} sx={{ p: 4, textAlign: 'center', borderRadius: 2 }}>
                      <Box sx={{ py: 4 }}>
                        <AnalyticsIcon sx={{ fontSize: 60, color: 'primary.light', mb: 2, opacity: 0.7 }} />
                        <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
                          {isStarting || isPolling ? 'Research in Progress' : 'Welcome to Research Intelligence Platform'}
                        </Typography>
                        <Typography color="text.secondary" sx={{ maxWidth: 600, mx: 'auto', mb: 3 }}>
                          {isStarting || isPolling 
                            ? 'Your comprehensive research report will appear here when ready...'
                            : 'Enter a research query to begin exploring data-driven insights with advanced analytics and visualizations.'}
                        </Typography>
                        {!isStarting && !isPolling && (
                          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
                            <Chip label="Market Analysis" color="primary" onClick={() => handleViewChange('research')} />
                            <Chip label="Competitive Research" color="secondary" onClick={() => handleViewChange('research')} />
                            <Chip label="Industry Trends" color="info" onClick={() => handleViewChange('research')} />
                            <Chip label="Technology Assessment" color="success" onClick={() => handleViewChange('research')} />
                          </Box>
                        )}
                      </Box>
                    </Paper>
                  )}
                </Grid>
              </Grid>
            )}
            
            {/* History View */}
            {activeView === 'history' && (
              <ProjectHistory onSelectProject={handleSelectHistoryProject} />
            )}
            
            {/* Settings View */}
            {activeView === 'settings' && (
              <Settings />
            )}
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;