import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  Alert,
  Snackbar,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Slider,
  Tabs,
  Tab,
  useTheme,
  useMediaQuery,
  CircularProgress,
  Chip
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Notifications as NotificationsIcon,
  Palette as PaletteIcon,
  Storage as StorageIcon,
  Security as SecurityIcon,
  Api as ApiIcon
} from '@mui/icons-material';

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index) {
  return {
    id: `settings-tab-${index}`,
    'aria-controls': `settings-tabpanel-${index}`,
  };
}

const Settings = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [tabValue, setTabValue] = useState(0);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  // General settings
  const [darkMode, setDarkMode] = useState(true);
  const [language, setLanguage] = useState('en');
  const [autoSave, setAutoSave] = useState(true);
  
  // API settings
  const [apiKey, setApiKey] = useState('');
  const [apiKeySet, setApiKeySet] = useState(false);
  const [apiModel, setApiModel] = useState('gemini-1.5-flash');
  const [temperature, setTemperature] = useState(0.3);
  
  // Storage settings
  const [maxProjects, setMaxProjects] = useState(50);
  const [storageLocation, setStorageLocation] = useState('local');
  
  // Notification settings
  const [emailNotifications, setEmailNotifications] = useState(false);
  const [email, setEmail] = useState('');
  
  // Loading state
  const [isLoading, setIsLoading] = useState(true);
  
  // Fetch settings from API
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        setIsLoading(true);
        const { getSettings } = await import('../services/api');
        const settings = await getSettings();
        
        // Update state with fetched settings
        if (settings.api_key) {
          setApiKey(settings.api_key);
          setApiKeySet(settings.api_key_set);
        }
        
        if (settings.api_model) {
          setApiModel(settings.api_model);
        }
        
        if (settings.temperature !== undefined) {
          setTemperature(settings.temperature);
        }
        
        if (settings.max_projects) {
          setMaxProjects(settings.max_projects);
        }
        
        if (settings.storage_location) {
          setStorageLocation(settings.storage_location);
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching settings:', error);
        setSnackbar({
          open: true,
          message: 'Error loading settings: ' + error.message,
          severity: 'error'
        });
        setIsLoading(false);
      }
    };
    
    fetchSettings();
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  const handleSaveSettings = async () => {
    try {
      setIsLoading(true);
      const { updateSettings } = await import('../services/api');
      
      // Prepare settings object
      const settings = {
        api_key: apiKey,
        api_model: apiModel,
        temperature: temperature,
        max_projects: maxProjects,
        storage_location: storageLocation,
        debug_mode: false // Not exposed in UI yet
      };
      
      // Send to API
      await updateSettings(settings);
      
      setSnackbar({
        open: true,
        message: 'Settings saved successfully!',
        severity: 'success'
      });
      
      // Update apiKeySet if a new key was provided
      if (apiKey && !apiKey.includes('*')) {
        setApiKeySet(true);
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error saving settings:', error);
      setSnackbar({
        open: true,
        message: 'Error saving settings: ' + error.message,
        severity: 'error'
      });
      setIsLoading(false);
    }
  };
  
  const handleResetSettings = async () => {
    try {
      // Reset to defaults
      setDarkMode(true);
      setLanguage('en');
      setAutoSave(true);
      setApiModel('gemini-1.5-flash');
      setTemperature(0.3);
      setMaxProjects(50);
      setStorageLocation('local');
      setEmailNotifications(false);
      setEmail('');
      
      // Don't reset API key, just refetch it
      const { getSettings } = await import('../services/api');
      const settings = await getSettings();
      if (settings.api_key) {
        setApiKey(settings.api_key);
        setApiKeySet(settings.api_key_set);
      }
      
      setSnackbar({
        open: true,
        message: 'Settings reset to defaults',
        severity: 'info'
      });
    } catch (error) {
      console.error('Error resetting settings:', error);
      setSnackbar({
        open: true,
        message: 'Error resetting settings: ' + error.message,
        severity: 'error'
      });
    }
  };
  
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={3} sx={{ p: 3, mb: 3, borderRadius: 2 }}>
        <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', mb: 3, display: 'flex', alignItems: 'center' }}>
          <SecurityIcon sx={{ mr: 1 }} />
          Settings
        </Typography>
        
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="settings tabs"
            variant={isMobile ? "scrollable" : "standard"}
            scrollButtons={isMobile ? "auto" : "standard"}
            allowScrollButtonsMobile
            centered={!isMobile}
          >
            <Tab icon={<PaletteIcon />} label="General" {...a11yProps(0)} />
            <Tab icon={<ApiIcon />} label="API" {...a11yProps(1)} />
            <Tab icon={<StorageIcon />} label="Storage" {...a11yProps(2)} />
            <Tab icon={<NotificationsIcon />} label="Notifications" {...a11yProps(3)} />
          </Tabs>
        </Box>
        
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Appearance
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={darkMode} 
                        onChange={(e) => setDarkMode(e.target.checked)} 
                      />
                    }
                    label="Dark Mode"
                  />
                  
                  <Box sx={{ mt: 3 }}>
                    <FormControl fullWidth>
                      <InputLabel id="language-select-label">Language</InputLabel>
                      <Select
                        labelId="language-select-label"
                        value={language}
                        label="Language"
                        onChange={(e) => setLanguage(e.target.value)}
                      >
                        <MenuItem value="en">English</MenuItem>
                        <MenuItem value="es">Spanish</MenuItem>
                        <MenuItem value="fr">French</MenuItem>
                        <MenuItem value="de">German</MenuItem>
                        <MenuItem value="zh">Chinese</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Behavior
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={autoSave} 
                        onChange={(e) => setAutoSave(e.target.checked)} 
                      />
                    }
                    label="Auto-save research projects"
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    LLM Configuration
                  </Typography>
                  
                  <TextField
                    fullWidth
                    label="API Key"
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    margin="normal"
                    helperText={apiKeySet ? "API key is set and valid" : "Enter your Google Gemini API key"}
                    disabled={isLoading}
                    color={apiKeySet ? "success" : "primary"}
                    InputProps={{
                      endAdornment: apiKeySet && (
                        <Chip 
                          size="small" 
                          color="success" 
                          label="Valid" 
                          sx={{ mr: 1 }}
                        />
                      )
                    }}
                  />
                  
                  <FormControl fullWidth margin="normal">
                    <InputLabel id="model-select-label">Model</InputLabel>
                    <Select
                      labelId="model-select-label"
                      value={apiModel}
                      label="Model"
                      onChange={(e) => setApiModel(e.target.value)}
                    >
                      <MenuItem value="gemini-1.5-flash">Gemini 1.5 Flash</MenuItem>
                      <MenuItem value="gemini-1.5-pro">Gemini 1.5 Pro</MenuItem>
                      <MenuItem value="gemini-1.0-pro">Gemini 1.0 Pro</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <Box sx={{ mt: 3 }}>
                    <Typography gutterBottom>
                      Temperature: {temperature}
                    </Typography>
                    <Slider
                      value={temperature}
                      min={0}
                      max={1}
                      step={0.1}
                      onChange={(e, newValue) => setTemperature(newValue)}
                      valueLabelDisplay="auto"
                    />
                    <Typography variant="caption" color="text.secondary">
                      Lower values produce more focused, deterministic outputs. Higher values produce more creative, varied outputs.
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Project Storage
                  </Typography>
                  
                  <FormControl fullWidth margin="normal">
                    <InputLabel id="storage-select-label">Storage Location</InputLabel>
                    <Select
                      labelId="storage-select-label"
                      value={storageLocation}
                      label="Storage Location"
                      onChange={(e) => setStorageLocation(e.target.value)}
                    >
                      <MenuItem value="local">Local SQLite Database</MenuItem>
                      <MenuItem value="cloud" disabled>Cloud Storage (Coming Soon)</MenuItem>
                    </Select>
                  </FormControl>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Storage Limits
                  </Typography>
                  
                  <TextField
                    fullWidth
                    label="Maximum Projects"
                    type="number"
                    value={maxProjects}
                    onChange={(e) => setMaxProjects(parseInt(e.target.value))}
                    margin="normal"
                    InputProps={{ inputProps: { min: 1, max: 100 } }}
                    helperText="Maximum number of research projects to store"
                  />
                  
                  <Box sx={{ mt: 3 }}>
                    <Button 
                      variant="outlined" 
                      color="error" 
                      startIcon={<RefreshIcon />}
                    >
                      Clear All Projects
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Email Notifications
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={emailNotifications} 
                        onChange={(e) => setEmailNotifications(e.target.checked)} 
                      />
                    }
                    label="Enable email notifications"
                  />
                  
                  {emailNotifications && (
                    <TextField
                      fullWidth
                      label="Email Address"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      margin="normal"
                      helperText="We'll send you notifications when your research is complete"
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'space-between' }}>
          <Button 
            variant="outlined" 
            color="secondary" 
            startIcon={<RefreshIcon />}
            onClick={handleResetSettings}
            disabled={isLoading}
          >
            Reset to Defaults
          </Button>
          
          <Button 
            variant="contained" 
            color="primary" 
            startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />}
            onClick={handleSaveSettings}
            disabled={isLoading}
          >
            {isLoading ? 'Saving...' : 'Save Settings'}
          </Button>
        </Box>
      </Paper>
      
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Settings;