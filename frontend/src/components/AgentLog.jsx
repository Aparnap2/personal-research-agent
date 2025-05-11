import React, { useEffect, useRef } from 'react';
import { Box, Typography, Paper, List, ListItem, ListItemText, ListItemIcon, Chip } from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import AutorenewIcon from '@mui/icons-material/Autorenew';

function AgentLog({ messages }) {
  const endOfMessagesRef = useRef(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (!messages || messages.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
        Waiting for agent updates...
      </Typography>
    );
  }

  const getIconAndColor = (type) => {
    switch (type?.toLowerCase()) {
      case 'error':
        return { icon: <ErrorOutlineIcon fontSize="small" />, color: 'error.main' };
      case 'success':
        return { icon: <CheckCircleOutlineIcon fontSize="small" />, color: 'success.main' };
      case 'info':
      case 'system':
        return { icon: <InfoOutlinedIcon fontSize="small" />, color: 'info.main' };
      case 'warning':
        return { icon: <AutorenewIcon fontSize="small" />, color: 'warning.main' };
      default:
        return { icon: <InfoOutlinedIcon fontSize="small" />, color: 'text.secondary' };
    }
  };

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        maxHeight: '400px', 
        overflowY: 'auto', 
        bgcolor: 'grey.50', 
        borderRadius: 2, 
        p: 1 
      }}
    >
      <List dense>
        {messages.map((msgObj, index) => {
          const { icon, color } = getIconAndColor(msgObj.type);
          const timePart = msgObj.text.match(/^\[(\d{2}:\d{2}:\d{2})\]/);
          const messageText = timePart ? msgObj.text.substring(timePart[0].length).trim() : msgObj.text;

          return (
            <ListItem key={index} sx={{ py: 0.5, alignItems: 'flex-start' }}>
              <ListItemIcon sx={{ minWidth: '32px', mt: 0.5, color }}>
                {icon}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
                    {msgObj.timestamp && (
                      <Chip
                        label={new Date(msgObj.timestamp).toLocaleTimeString()}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 1, fontSize: '0.75rem', height: 20 }}
                      />
                    )}
                    <Typography 
                      variant="body2" 
                      sx={{ wordBreak: 'break-word', color: 'text.primary' }}
                    >
                      {messageText}
                    </Typography>
                  </Box>
                }
              />
            </ListItem>
          );
        })}
        <div ref={endOfMessagesRef} />
      </List>
    </Paper>
  );
}

export default AgentLog;