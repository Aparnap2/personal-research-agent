// frontend/src/components/AgentLog.jsx
import React, { useEffect, useRef } from 'react';
import { Box, Typography, Paper, List, ListItem, ListItemText, ListItemIcon, Chip } from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import AutorenewIcon from '@mui/icons-material/Autorenew'; // For processing steps

function AgentLog({ messages }) {
  const endOfMessagesRef = useRef(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (!messages || messages.length === 0) {
    return <Typography variant="body2" color="textSecondary" sx={{p:1}}>Waiting for agent updates...</Typography>;
  }

  const getIconAndColor = (text) => {
    const lowerText = text.toLowerCase();
    if (lowerText.includes("error:") || lowerText.includes("failed:")) return { icon: <ErrorOutlineIcon fontSize="inherit" />, color: 'error.main' };
    if (lowerText.includes("success:") || lowerText.includes("completed") || lowerText.includes("generated")) return { icon: <CheckCircleOutlineIcon fontSize="inherit" />, color: 'success.main' };
    if (lowerText.includes("warning:")) return { icon: <InfoOutlinedIcon fontSize="inherit" />, color: 'warning.main' };
    if (lowerText.includes("starting:") || lowerText.includes("scraping url") || lowerText.includes("processing")) return { icon: <AutorenewIcon fontSize="inherit" />, color: 'info.main' };
    return { icon: <InfoOutlinedIcon fontSize="inherit" />, color: 'text.secondary' };
  };

  return (
    <Paper elevation={0} sx={{ maxHeight: '300px', overflowY: 'auto', mt: 1, p: 0.5, backgroundColor: 'transparent' }}>
      <List dense disablePadding>
        {messages.map((msgObj, index) => {
          const { icon, color } = getIconAndColor(msgObj.text);
          const timePart = msgObj.text.match(/^\[(\d{2}:\d{2}:\d{2})\]/);
          const messageText = timePart ? msgObj.text.substring(timePart[0].length).trim() : msgObj.text;

          return (
            <ListItem key={index} disableGutters sx={{py: 0.3, alignItems: 'flex-start'}}>
              <ListItemIcon sx={{minWidth: '30px', mt: '4px', color: color}}>
                {icon}
              </ListItemIcon>
              <ListItemText 
                primary={
                    <Typography variant="body2" component="span" sx={{wordBreak: 'break-word'}}>
                        {timePart && <Chip label={timePart[1]} size="small" variant="outlined" sx={{mr:0.5, height: '18px', fontSize: '0.7rem'}}/>}
                        {messageText}
                    </Typography>
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