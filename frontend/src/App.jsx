import React, { useState, useCallback } from 'react';
import {
  ThemeProvider,
  CssBaseline,
  Box,
  Typography,
  AppBar,
  Toolbar,
  IconButton,
  Tooltip,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Tabs,
  Tab,
  useMediaQuery,
  Chip,
  Paper,
  Alert
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
import ViewListIcon from '@mui/icons-material/ViewList';
import MapIcon from '@mui/icons-material/Map';
import ChatIcon from '@mui/icons-material/Chat';
import ArticleIcon from '@mui/icons-material/Article';
import ChatInterface from './components/ChatInterface';
import ResearchMindmap from './components/ResearchMindmap';
import ReportDisplay from './components/ReportDisplay';
import ProjectHistory from './components/ProjectHistory';
import Dashboard from './components/DashboardWithRealData';
import Settings from './components/Settings';
import { startResearch, getResearchStatus, getSubQuestions, getResearchReport } from './services/api';
import theme from './theme';

const DRAWER_WIDTH = 240;
const RIGHT_PANEL_WIDTH = 400;

function App() {
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  // Navigation state
  const [drawerOpen, setDrawerOpen] = useState(!isMobile);
  const [activeView, setActiveView] = useState('research'); // 'research', 'history', 'dashboard', 'settings'

  // Research state
  const [projectId, setProjectId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Chat state
  const [messages, setMessages] = useState([]);
  const [subQuestions, setSubQuestions] = useState([]);
  const [reportData, setReportData] = useState(null);

  // Right panel state
  const [rightPanelTab, setRightPanelTab] = useState(0); // 0: Mindmap, 1: Report, 2: Sources
  const [rightPanelOpen, setRightPanelOpen] = useState(true);

  const toggleDrawer = () => setDrawerOpen(!drawerOpen);

  const handleViewChange = (view) => {
    setActiveView(view);
    if (isMobile) setDrawerOpen(false);
  };

  // Load sub-questions and report when project is selected
  const loadProjectData = useCallback(async (pid) => {
    try {
      const [subQData, report] = await Promise.all([
        getSubQuestions(pid).catch(() => ({ sub_questions: [] })),
        getResearchReport(pid).catch(() => null)
      ]);
      setSubQuestions(subQData.sub_questions || []);
      if (report) setReportData(report);
    } catch (err) {
      console.error('Error loading project data:', err);
    }
  }, []);

  // Handle starting a new research
  const handleSendMessage = async (message) => {
    if (!message.trim()) return;

    // Add user message
    const userMsg = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);
    setError('');

    try {
      // Start research
      const response = await startResearch(message);
      const newProjectId = response.project_id;
      setProjectId(newProjectId);

      // Add system message
      setMessages(prev => [...prev, {
        role: 'system',
        title: 'Research Started',
        content: `Research initialized with ID: ${newProjectId.slice(0, 8)}...`,
        timestamp: new Date().toISOString()
      }]);

      // Poll for status updates
      await pollResearchStatus(newProjectId);

    } catch (err) {
      console.error('Research error:', err);
      setError(err.message || 'Failed to start research');
      setMessages(prev => [...prev, {
        role: 'system',
        title: 'Error',
        content: err.message,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Poll research status until complete
  const pollResearchStatus = async (pid) => {
    const maxAttempts = 120; // 6 minutes max
    let attempts = 0;

    const poll = async () => {
      attempts++;
      if (attempts > maxAttempts) {
        setError('Research timed out');
        return;
      }

      try {
        const status = await getResearchStatus(pid);

        // Update messages based on status
        if (status.messages && status.messages.length > 0) {
          const latestMsg = status.messages[status.messages.length - 1];
          setMessages(prev => {
            const filtered = prev.filter(p =>
              !(p.role === 'assistant' && p.title === 'Thinking')
            );
            return [...filtered, {
              role: 'assistant',
              title: latestMsg.type === 'error' ? 'Error' : 'Research Update',
              content: latestMsg.text || latestMsg.content || '',
              timestamp: latestMsg.timestamp || new Date().toISOString()
            }];
          });
        }

        // Update sub-questions if available
        if (status.sub_questions) {
          setSubQuestions(status.sub_questions);
        }

        if (status.completed) {
          // Load final report
          const report = await getResearchReport(pid).catch(() => null);
          if (report) {
            setReportData(report);
            setSubQuestions(report.sub_questions || []);
          }
          setMessages(prev => [...prev, {
            role: 'assistant',
            title: 'Complete',
            content: 'Research completed successfully!',
            timestamp: new Date().toISOString()
          }]);
        } else {
          // Continue polling
          setTimeout(poll, 2000);
        }
      } catch (err) {
        console.error('Polling error:', err);
        if (attempts < maxAttempts) {
          setTimeout(poll, 3000);
        } else {
          setError('Failed to get research updates');
        }
      }
    };

    poll();
  };

  // Handle node click in mindmap
  const handleNodeClick = (node, subQuestion) => {
    if (subQuestion) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        title: 'Question Details',
        content: subQuestion.question || node.data?.label,
        timestamp: new Date().toISOString()
      }]);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        {/* App Bar */}
        <AppBar position="static" sx={{ bgcolor: 'primary.dark' }}>
          <Toolbar>
            <IconButton
              color="inherit"
              edge="start"
              onClick={toggleDrawer}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <AnalyticsIcon sx={{ mr: 2 }} />
            <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
              Research Intelligence Platform
            </Typography>
            <Tooltip title="View on GitHub">
              <IconButton color="inherit" component="a"
                href="https://github.com/Aparnap2/personal-research-agent" target="_blank">
                <GitHubIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Help">
              <IconButton color="inherit">
                <HelpOutlineIcon />
              </IconButton>
            </Tooltip>
          </Toolbar>
        </AppBar>

        <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Left Navigation Drawer */}
          <Drawer
            variant={isMobile ? "temporary" : "persistent"}
            open={isMobile ? drawerOpen : true}
            onClose={toggleDrawer}
            sx={{
              width: DRAWER_WIDTH,
              flexShrink: 0,
              '& .MuiDrawer-paper': {
                width: DRAWER_WIDTH,
                boxSizing: 'border-box',
                top: isMobile ? 0 : '64px',
                height: isMobile ? '100%' : 'calc(100% - 64px)',
              },
            }}
          >
            <List>
              <ListItem button selected={activeView === 'dashboard'} onClick={() => handleViewChange('dashboard')}>
                <ListItemIcon><DashboardIcon color={activeView === 'dashboard' ? 'primary' : 'inherit'} /></ListItemIcon>
                <ListItemText primary="Dashboard" />
              </ListItem>
              <ListItem button selected={activeView === 'research'} onClick={() => handleViewChange('research')}>
                <ListItemIcon><SearchIcon color={activeView === 'research' ? 'primary' : 'inherit'} /></ListItemIcon>
                <ListItemText primary="Research" />
              </ListItem>
              <ListItem button selected={activeView === 'history'} onClick={() => handleViewChange('history')}>
                <ListItemIcon><HistoryIcon color={activeView === 'history' ? 'primary' : 'inherit'} /></ListItemIcon>
                <ListItemText primary="History" />
              </ListItem>
              <Divider />
              <ListItem button selected={activeView === 'settings'} onClick={() => handleViewChange('settings')}>
                <ListItemIcon><SettingsIcon color={activeView === 'settings' ? 'primary' : 'inherit'} /></ListItemIcon>
                <ListItemText primary="Settings" />
              </ListItem>
            </List>
          </Drawer>

          {/* Main Content */}
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              display: 'flex',
              flexDirection: 'column',
              ml: isMobile ? 0 : `${DRAWER_WIDTH}px`,
              transition: theme.transitions.create('margin', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.leavingScreen,
              }),
            }}
          >
            {/* Error Alert */}
            {error && (
              <Alert
                severity="error"
                sx={{ m: 2, borderRadius: 2 }}
                action={
                  <IconButton size="small" color="inherit" onClick={() => setError('')}>
                    <CloseIcon fontSize="inherit" />
                  </IconButton>
                }
              >
                {error}
              </Alert>
            )}

            {/* Dashboard View */}
            {activeView === 'dashboard' && <Dashboard />}

            {/* Research View - Chat-First Layout */}
            {activeView === 'research' && (
              <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                {/* Left Panel: Chat Interface */}
                <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                  <ChatInterface
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    isLoading={isLoading}
                    projectId={projectId}
                  />
                </Box>

                {/* Right Panel: Mindmap / Report Tabs */}
                {rightPanelOpen && (
                  <Box
                    sx={{
                      width: RIGHT_PANEL_WIDTH,
                      borderLeft: `1px solid ${theme.palette.divider}`,
                      display: 'flex',
                      flexDirection: 'column',
                      bgcolor: 'background.paper',
                    }}
                  >
                    {/* Tab Bar */}
                    <Tabs
                      value={rightPanelTab}
                      onChange={(e, v) => setRightPanelTab(v)}
                      variant="fullWidth"
                      sx={{ borderBottom: `1px solid ${theme.palette.divider}` }}
                    >
                      <Tab icon={<MapIcon />} label="Mindmap" iconPosition="start" />
                      <Tab icon={<ArticleIcon />} label="Report" iconPosition="start" disabled={!reportData} />
                      <Tab icon={<ViewListIcon />} label="Sources" iconPosition="start" disabled={!reportData} />
                    </Tabs>

                    {/* Tab Content */}
                    <Box sx={{ flex: 1, overflow: 'auto' }}>
                      {rightPanelTab === 0 && (
                        <ResearchMindmap
                          subQuestions={subQuestions}
                          onNodeClick={handleNodeClick}
                          projectId={projectId}
                        />
                      )}
                      {rightPanelTab === 1 && reportData && (
                        <ReportDisplay reportData={reportData} />
                      )}
                      {rightPanelTab === 2 && reportData && (
                        <Box sx={{ p: 2 }}>
                          <Typography variant="subtitle2" gutterBottom>Sources</Typography>
                          {reportData.sources?.map((source, idx) => (
                            <Chip
                              key={idx}
                              label={source.slice(0, 40) + '...'}
                              size="small"
                              variant="outlined"
                              sx={{ m: 0.5 }}
                            />
                          ))}
                        </Box>
                      )}
                    </Box>
                  </Box>
                )}

                {/* Toggle Right Panel Button */}
                <IconButton
                  onClick={() => setRightPanelOpen(!rightPanelOpen)}
                  sx={{
                    position: 'absolute',
                    right: rightPanelOpen ? RIGHT_PANEL_WIDTH + 8 : 8,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    bgcolor: 'background.paper',
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: '50%',
                    width: 28,
                    height: 28,
                    zIndex: 10,
                    '&:hover': { bgcolor: 'action.hover' }
                  }}
                  size="small"
                >
                  {rightPanelOpen ? '»' : '«'}
                </IconButton>
              </Box>
            )}

            {/* History View */}
            {activeView === 'history' && (
              <ProjectHistory
                onSelectProject={(pid) => {
                  setProjectId(pid);
                  setActiveView('research');
                  loadProjectData(pid);
                }}
              />
            )}

            {/* Settings View */}
            {activeView === 'settings' && <Settings />}
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
