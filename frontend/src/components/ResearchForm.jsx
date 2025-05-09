import React, { useState, useEffect } from 'react';
import { 
  TextField, 
  Button, 
  Box, 
  CircularProgress,
  IconButton,
  Tooltip,
  Collapse,
  Alert,
  Typography,
  Paper,
  Divider,
  Chip,
  Stack
} from '@mui/material';
import { 
  Send as SendIcon,
  History as HistoryIcon,
  Clear as ClearIcon,
  HelpOutline as HelpIcon,
  ContentPaste as PasteIcon
} from '@mui/icons-material';

const exampleQueries = [
  "Analyze current trends in AI-driven marketing automation for SaaS companies",
  "Compare the effectiveness of different machine learning models for image recognition",
  "Research the latest developments in quantum computing and potential business applications",
  "Investigate the impact of blockchain technology on supply chain management",
  "Summarize recent breakthroughs in renewable energy storage solutions"
];

function ResearchForm({ onSubmit, isLoading, recentQueries = [] }) {
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const [showExamples, setShowExamples] = useState(false);
  const [charCount, setCharCount] = useState(0);
  const maxChars = 2000;

  useEffect(() => {
    setCharCount(query.length);
  }, [query]);

  const handleSubmit = (event) => {
    event.preventDefault();
    setError('');
    
    if (!query.trim()) {
      setError('Please enter a research query');
      return;
    }
    
    if (query.length > maxChars) {
      setError(`Query exceeds maximum length of ${maxChars} characters`);
      return;
    }
    
    onSubmit(query);
  };

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text.length > maxChars) {
        setError(`Pasted content exceeds maximum length of ${maxChars} characters`);
        return;
      }
      setQuery(text);
    } catch (err) {
      setError('Failed to access clipboard. Please paste manually.');
    }
  };

  const handleExampleSelect = (example) => {
    setQuery(example);
    setShowExamples(false);
  };

  const handleClear = () => {
    setQuery('');
    setError('');
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
      {/* Error Alert */}
      <Collapse in={!!error}>
        <Alert 
          severity="error" 
          sx={{ mb: 2 }}
          action={
            <IconButton
              size="small"
              onClick={() => setError('')}
            >
              <ClearIcon fontSize="inherit" />
            </IconButton>
          }
        >
          {error}
        </Alert>
      </Collapse>

      {/* Query Input */}
      <TextField
        margin="normal"
        required
        fullWidth
        id="research-query"
        label="Your Research Query"
        name="query"
        autoFocus
        multiline
        minRows={4}
        maxRows={8}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={isLoading}
        placeholder="e.g., Analyze current trends in AI-driven marketing automation..."
        InputProps={{
          endAdornment: (
            <Box sx={{ position: 'absolute', right: 8, bottom: 8 }}>
              <Typography 
                variant="caption" 
                color={charCount > maxChars ? 'error' : 'text.secondary'}
              >
                {charCount}/{maxChars}
              </Typography>
            </Box>
          )
        }}
      />

      {/* Action Buttons */}
      <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
        <Button
          type="submit"
          variant="contained"
          sx={{ flex: 1, py: 1.5 }}
          disabled={isLoading || !query.trim()}
          endIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
        >
          {isLoading ? 'Processing...' : 'Start Research'}
        </Button>

        <Tooltip title="Clear query">
          <IconButton
            onClick={handleClear}
            disabled={!query.trim() || isLoading}
            color="secondary"
          >
            <ClearIcon />
          </IconButton>
        </Tooltip>

        <Tooltip title="Paste from clipboard">
          <IconButton
            onClick={handlePaste}
            disabled={isLoading}
            color="primary"
          >
            <PasteIcon />
          </IconButton>
        </Tooltip>
      </Stack>

      {/* Example Queries */}
      <Box sx={{ mb: 2 }}>
        <Button
          size="small"
          startIcon={<HelpIcon />}
          onClick={() => setShowExamples(!showExamples)}
          sx={{ textTransform: 'none' }}
        >
          {showExamples ? 'Hide examples' : 'Need inspiration?'}
        </Button>
        
        <Collapse in={showExamples}>
          <Paper elevation={0} sx={{ p: 2, mt: 1, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" gutterBottom>
              Example Research Queries:
            </Typography>
            <Stack direction="column" spacing={1}>
              {exampleQueries.map((example, index) => (
                <Chip
                  key={index}
                  label={example}
                  onClick={() => handleExampleSelect(example)}
                  sx={{ 
                    justifyContent: 'flex-start',
                    height: 'auto',
                    py: 1,
                    textAlign: 'left',
                    whiteSpace: 'normal'
                  }}
                />
              ))}
            </Stack>
          </Paper>
        </Collapse>
      </Box>

      {/* Recent Queries (if available) */}
      {recentQueries.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Recent Queries:
          </Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap' }}>
            {recentQueries.slice(0, 3).map((recent, index) => (
              <Chip
                key={index}
                label={recent}
                onClick={() => setQuery(recent)}
                icon={<HistoryIcon />}
                size="small"
                variant="outlined"
              />
            ))}
          </Stack>
        </Box>
      )}
    </Box>
  );
}

export default React.memo(ResearchForm);