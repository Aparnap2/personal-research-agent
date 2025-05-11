import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemSecondaryAction,
  IconButton, 
  Divider, 
  Chip,
  Tooltip,
  CircularProgress,
  Alert,
  Button
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import ShareIcon from '@mui/icons-material/Share';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import { format } from 'date-fns';

// This would be replaced with actual API calls in a real implementation
const mockProjects = [
  { 
    id: 'research_12345', 
    query: 'Market analysis of renewable energy in Europe 2024', 
    timestamp: '2024-05-15T14:30:00Z',
    status: 'completed',
    charts: 5,
    wordCount: 4250
  },
  { 
    id: 'research_23456', 
    query: 'AI adoption in healthcare diagnostics', 
    timestamp: '2024-05-10T09:15:00Z',
    status: 'completed',
    charts: 3,
    wordCount: 3800
  },
  { 
    id: 'research_34567', 
    query: 'Consumer trends in sustainable fashion', 
    timestamp: '2024-05-05T16:45:00Z',
    status: 'completed',
    charts: 4,
    wordCount: 4100
  }
];

function ProjectHistory({ onSelectProject }) {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Simulate API call
    const fetchProjects = async () => {
      try {
        setLoading(true);
        // In a real implementation, this would be an API call
        // const response = await getProjectHistory();
        // setProjects(response.data);
        
        // Using mock data for now
        setTimeout(() => {
          setProjects(mockProjects);
          setLoading(false);
        }, 1000);
      } catch (err) {
        setError('Failed to load project history');
        setLoading(false);
      }
    };

    fetchProjects();
  }, []);

  const handleViewProject = (projectId) => {
    if (onSelectProject) {
      onSelectProject(projectId);
    }
  };

  const handleDeleteProject = (projectId, event) => {
    event.stopPropagation();
    // This would call an API to delete the project
    setProjects(projects.filter(p => p.id !== projectId));
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'primary.main', display: 'flex', alignItems: 'center' }}>
          <FolderOpenIcon sx={{ mr: 1 }} />
          Research Project History
        </Typography>
        <Button variant="outlined" size="small">
          Clear All
        </Button>
      </Box>
      
      {projects.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
          No research projects found. Start a new research to see your history.
        </Typography>
      ) : (
        <List sx={{ width: '100%' }}>
          {projects.map((project, index) => (
            <React.Fragment key={project.id}>
              {index > 0 && <Divider component="li" />}
              <ListItem 
                alignItems="flex-start" 
                sx={{ 
                  cursor: 'pointer',
                  '&:hover': { bgcolor: 'action.hover' }
                }}
                onClick={() => handleViewProject(project.id)}
              >
                <ListItemText
                  primary={
                    <Typography variant="subtitle1" sx={{ fontWeight: 'medium' }}>
                      {project.query}
                    </Typography>
                  }
                  secondary={
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2" color="text.secondary" component="span">
                        {format(new Date(project.timestamp), 'PPP p')}
                      </Typography>
                      <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                        <Chip 
                          size="small" 
                          label={`${project.charts} charts`} 
                          color="primary" 
                          variant="outlined"
                        />
                        <Chip 
                          size="small" 
                          label={`${project.wordCount} words`} 
                          color="secondary" 
                          variant="outlined"
                        />
                        <Chip 
                          size="small" 
                          label={project.status} 
                          color={project.status === 'completed' ? 'success' : 'warning'} 
                          variant="outlined"
                        />
                      </Box>
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Box sx={{ display: 'flex' }}>
                    <Tooltip title="View Report">
                      <IconButton edge="end" onClick={() => handleViewProject(project.id)}>
                        <VisibilityIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Download">
                      <IconButton edge="end">
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton edge="end" onClick={(e) => handleDeleteProject(project.id, e)}>
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </ListItemSecondaryAction>
              </ListItem>
            </React.Fragment>
          ))}
        </List>
      )}
    </Paper>
  );
}

export default ProjectHistory;