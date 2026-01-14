import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  Avatar,
  Chip,
  CircularProgress,
  Divider,
  useTheme
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';

const MESSAGE_TYPES = {
  USER: 'user',
  ASSISTANT: 'assistant',
  TOOL: 'tool',
  SYSTEM: 'system'
};

function MessageBubble({ message }) {
  const theme = useTheme();
  const isUser = message.role === MESSAGE_TYPES.USER;
  const isTool = message.role === MESSAGE_TYPES.TOOL;

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 1.5,
        px: 2
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: isUser ? 'row-reverse' : 'row',
          alignItems: 'flex-start',
          maxWidth: '85%'
        }}
      >
        <Avatar
          sx={{
            bgcolor: isUser ? theme.palette.primary.main :
                     isTool ? theme.palette.warning.main :
                     theme.palette.secondary.main,
            width: 32,
            height: 32
          }}
        >
          {isUser ? <PersonIcon fontSize="small" /> :
           isTool ? <AutoAwesomeIcon fontSize="small" /> :
           <SmartToyIcon fontSize="small" />}
        </Avatar>

        <Paper
          elevation={0}
          sx={{
            p: 1.5,
            ml: isUser ? 0 : 1,
            mr: isUser ? 1 : 0,
            bgcolor: isUser ? theme.palette.primary.light :
                       isTool ? theme.palette.warning.light :
                       theme.palette.grey[100],
            borderRadius: 2,
            borderTopLeftRadius: !isUser ? 0 : undefined,
            borderTopRightRadius: isUser ? 0 : undefined
          }}
        >
          {message.title && (
            <Chip
              label={message.title}
              size="small"
              sx={{ mb: 0.5, height: 20, fontSize: '0.7rem' }}
              color={isTool ? 'warning' : 'default'}
            />
          )}
          <Typography
            variant="body2"
            sx={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              color: theme.palette.text.primary
            }}
          >
            {message.content}
          </Typography>
          {message.timestamp && (
            <Typography
              variant="caption"
              sx={{
                display: 'block',
                mt: 0.5,
                opacity: 0.6,
                textAlign: isUser ? 'right' : 'left'
              }}
            >
              {new Date(message.timestamp).toLocaleTimeString()}
            </Typography>
          )}
        </Paper>
      </Box>
    </Box>
  );
}

function ChatInterface({
  messages = [],
  onSendMessage,
  isLoading = false,
  projectId = null
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const theme = useTheme();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const suggestedPrompts = [
    "Analyze market trends for AI",
    "Research renewable energy stats",
    "Compare programming language popularity",
    "Find healthcare statistics"
  ];

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        bgcolor: theme.palette.background.default
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: `1px solid ${theme.palette.divider}`,
          display: 'flex',
          alignItems: 'center',
          gap: 1
        }}
      >
        <SmartToyIcon color="primary" />
        <Typography variant="h6">
          Research Assistant
        </Typography>
        {projectId && (
          <Chip
            label={`Project: ${projectId.slice(0, 8)}...`}
            size="small"
            variant="outlined"
          />
        )}
      </Box>

      {/* Messages Area */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          py: 2,
          '&::-webkit-scrollbar': {
            width: '6px'
          },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: theme.palette.divider,
            borderRadius: '3px'
          }
        }}
      >
        {messages.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              Start a research conversation
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Ask me anything about research topics, and I'll help you find and analyze information.
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, justifyContent: 'center' }}>
              {suggestedPrompts.map((prompt, index) => (
                <Chip
                  key={index}
                  label={prompt}
                  onClick={() => setInput(prompt)}
                  variant="outlined"
                  size="small"
                  sx={{ cursor: 'pointer' }}
                />
              ))}
            </Box>
          </Box>
        ) : (
          messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
            />
          ))
        )}

        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'flex-start', px: 2, mb: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Researching...
              </Typography>
            </Box>
          </Box>
        )}

        <div ref={messagesEndRef} />
      </Box>

      <Divider />

      {/* Input Area */}
      <Box
        sx={{
          p: 2,
          borderTop: `1px solid ${theme.palette.divider}`,
          bgcolor: theme.palette.background.paper
        }}
      >
        <Box
          sx={{
            display: 'flex',
            gap: 1,
            alignItems: 'flex-end'
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={4}
            placeholder="Ask a research question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            size="small"
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2
              }
            }}
          />
          <IconButton
            color="primary"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            sx={{
              bgcolor: input.trim() && !isLoading ? 'primary.main' : 'grey.300',
              color: 'white',
              '&:hover': {
                bgcolor: input.trim() && !isLoading ? 'primary.dark' : 'grey.400'
              }
            }}
          >
            <SendIcon />
          </IconButton>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Press Enter to send, Shift+Enter for new line
        </Typography>
      </Box>
    </Box>
  );
}

export default ChatInterface;
