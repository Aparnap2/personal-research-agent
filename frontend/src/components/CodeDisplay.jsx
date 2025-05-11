import React, { useState } from 'react';
import { 
  Paper, Typography, Box, Button, IconButton, Tooltip, 
  Snackbar, Alert, useTheme, Divider
} from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CodeIcon from '@mui/icons-material/Code';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CodeDisplay = ({ code, title = "Generated Python Code" }) => {
  const theme = useTheme();
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  
  const handleCopyCode = () => {
    navigator.clipboard.writeText(code);
    setSnackbarOpen(true);
  };
  
  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };
  
  const toggleExpand = () => {
    setExpanded(!expanded);
  };
  
  // Determine if code is too long and should be truncated
  const isTooLong = code.split('\n').length > 20;
  const displayCode = !expanded && isTooLong 
    ? code.split('\n').slice(0, 20).join('\n') + '\n\n// ... (click "Show More" to see full code)'
    : code;
  
  return (
    <Paper elevation={3} sx={{ mt: 3, mb: 3, overflow: 'hidden', borderRadius: 2 }}>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        p: 2, 
        bgcolor: theme.palette.mode === 'dark' ? 'grey.800' : 'grey.100'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <CodeIcon sx={{ mr: 1, color: theme.palette.primary.main }} />
          <Typography variant="h6" component="h3">
            {title}
          </Typography>
        </Box>
        <Tooltip title="Copy code">
          <IconButton onClick={handleCopyCode} size="small">
            <ContentCopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      
      <Divider />
      
      <Box sx={{ maxHeight: expanded ? 'none' : '400px', overflow: 'auto' }}>
        <SyntaxHighlighter
          language="python"
          style={theme.palette.mode === 'dark' ? vscDarkPlus : vs}
          showLineNumbers
          wrapLines
          customStyle={{
            margin: 0,
            padding: '16px',
            borderRadius: 0,
            fontSize: '0.9rem',
          }}
        >
          {displayCode}
        </SyntaxHighlighter>
      </Box>
      
      {isTooLong && (
        <Box sx={{ 
          p: 1, 
          textAlign: 'center', 
          bgcolor: theme.palette.mode === 'dark' ? 'grey.800' : 'grey.100'
        }}>
          <Button 
            size="small" 
            onClick={toggleExpand}
            variant="text"
          >
            {expanded ? "Show Less" : "Show More"}
          </Button>
        </Box>
      )}
      
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="success" sx={{ width: '100%' }}>
          Code copied to clipboard!
        </Alert>
      </Snackbar>
    </Paper>
  );
};

export default CodeDisplay;