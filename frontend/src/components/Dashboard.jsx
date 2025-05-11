import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  Card, 
  CardContent, 
  CardHeader,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  LinearProgress,
  Chip
} from '@mui/material';
import BarChartIcon from '@mui/icons-material/BarChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AssessmentIcon from '@mui/icons-material/Assessment';
import DescriptionIcon from '@mui/icons-material/Description';
import SearchIcon from '@mui/icons-material/Search';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PendingIcon from '@mui/icons-material/Pending';
import ErrorIcon from '@mui/icons-material/Error';

// Mock data for dashboard
const mockStats = {
  totalProjects: 12,
  completedProjects: 9,
  inProgressProjects: 2,
  failedProjects: 1,
  totalCharts: 43,
  avgWordsPerReport: 4120,
  topQueries: [
    "Market analysis of renewable energy in Europe 2024",
    "AI adoption in healthcare diagnostics",
    "Consumer trends in sustainable fashion"
  ],
  recentActivity: [
    { action: "Research Completed", project: "Market analysis of renewable energy", timestamp: "2 hours ago" },
    { action: "Research Started", project: "Impact of AI on job market", timestamp: "5 hours ago" },
    { action: "Chart Generated", project: "Consumer trends in sustainable fashion", timestamp: "1 day ago" }
  ]
};

function Dashboard() {
  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 'bold', color: 'primary.main' }}>
        Research Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={2} sx={{ borderRadius: 2, height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <DescriptionIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Total Projects
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 'medium' }}>
                {mockStats.totalProjects}
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">Completed</Typography>
                  <Typography variant="caption" color="success.main">{mockStats.completedProjects}</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={(mockStats.completedProjects / mockStats.totalProjects) * 100} 
                  color="success"
                  sx={{ height: 6, borderRadius: 3, mb: 1 }}
                />
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">In Progress</Typography>
                  <Typography variant="caption" color="info.main">{mockStats.inProgressProjects}</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={(mockStats.inProgressProjects / mockStats.totalProjects) * 100} 
                  color="info"
                  sx={{ height: 6, borderRadius: 3, mb: 1 }}
                />
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">Failed</Typography>
                  <Typography variant="caption" color="error.main">{mockStats.failedProjects}</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={(mockStats.failedProjects / mockStats.totalProjects) * 100} 
                  color="error"
                  sx={{ height: 6, borderRadius: 3 }}
                />
              </Box>
            </CardContent>
          </Paper>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={2} sx={{ borderRadius: 2, height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <BarChartIcon color="secondary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Visualizations
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 'medium' }}>
                {mockStats.totalCharts}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Charts generated across all projects
              </Typography>
              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Chip size="small" label="Bar Charts" color="primary" variant="outlined" />
                <Chip size="small" label="Line Charts" color="secondary" variant="outlined" />
                <Chip size="small" label="Pie Charts" color="info" variant="outlined" />
              </Box>
            </CardContent>
          </Paper>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={2} sx={{ borderRadius: 2, height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AssessmentIcon color="info" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Report Analytics
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 'medium' }}>
                {mockStats.avgWordsPerReport}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Average words per report
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">Comprehensiveness</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={85} 
                  color="success"
                  sx={{ height: 6, borderRadius: 3 }}
                />
              </Box>
            </CardContent>
          </Paper>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Paper elevation={2} sx={{ borderRadius: 2, height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUpIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Research Efficiency
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 'medium' }}>
                92%
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Success rate for research queries
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">Avg. Completion Time</Typography>
                  <Typography variant="caption" color="text.primary">4.2 min</Typography>
                </Box>
              </Box>
            </CardContent>
          </Paper>
        </Grid>
        
        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ borderRadius: 2 }}>
            <CardHeader 
              title={
                <Typography variant="h6" sx={{ fontWeight: 'medium' }}>
                  Recent Activity
                </Typography>
              }
            />
            <Divider />
            <List sx={{ p: 0 }}>
              {mockStats.recentActivity.map((activity, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <Divider component="li" variant="inset" />}
                  <ListItem>
                    <ListItemIcon>
                      {activity.action === "Research Completed" ? (
                        <CheckCircleIcon color="success" />
                      ) : activity.action === "Research Started" ? (
                        <PendingIcon color="info" />
                      ) : (
                        <BarChartIcon color="secondary" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={activity.action}
                      secondary={
                        <React.Fragment>
                          <Typography component="span" variant="body2" color="text.primary">
                            {activity.project}
                          </Typography>
                          {" — "}{activity.timestamp}
                        </React.Fragment>
                      }
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Grid>
        
        {/* Top Queries */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ borderRadius: 2 }}>
            <CardHeader 
              title={
                <Typography variant="h6" sx={{ fontWeight: 'medium' }}>
                  Top Research Queries
                </Typography>
              }
            />
            <Divider />
            <List sx={{ p: 0 }}>
              {mockStats.topQueries.map((query, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <Divider component="li" variant="inset" />}
                  <ListItem>
                    <ListItemIcon>
                      <SearchIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary={query}
                      secondary={`Query #${index + 1}`}
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;