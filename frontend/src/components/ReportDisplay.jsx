import React, { useState, useEffect } from 'react';
import { 
  Paper, Typography, Box, Tabs, Tab, Card, CardMedia, CardContent, 
  IconButton, Tooltip, Alert, Grid, Snackbar, Chip, Divider,
  Button, useTheme, useMediaQuery, Link
} from '@mui/material';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkFootnotes from 'remark-footnotes';
import remarkToc from 'remark-toc';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import FileCopyIcon from '@mui/icons-material/FileCopy';
import DownloadIcon from '@mui/icons-material/Download';
import PrintIcon from '@mui/icons-material/Print';
import ShareIcon from '@mui/icons-material/Share';
import BarChartIcon from '@mui/icons-material/BarChart';
import InsightsIcon from '@mui/icons-material/Insights';
import SummarizeIcon from '@mui/icons-material/Summarize';
import RecommendIcon from '@mui/icons-material/Recommend';
import DataUsageIcon from '@mui/icons-material/DataUsage';
import LinkIcon from '@mui/icons-material/Link';
import CodeIcon from '@mui/icons-material/Code';
import CodeDisplay from './CodeDisplay';

// Custom image renderer with improved styling
const ImgComponent = ({ node, ...props }) => (
  <Box sx={{ 
    display: 'flex', 
    flexDirection: 'column', 
    alignItems: 'center',
    my: 3
  }}>
    <img 
      {...props} 
      style={{ 
        maxWidth: '100%', 
        height: 'auto', 
        borderRadius: '8px', 
        boxShadow: '0 4px 16px rgba(0,0,0,0.1)' 
      }} 
      alt={props.alt || 'Chart or Media'} 
    />
    {props.alt && (
      <Typography variant="caption" sx={{ mt: 1, fontStyle: 'italic', textAlign: 'center', maxWidth: '90%' }}>
        {props.alt}
      </Typography>
    )}
  </Box>
);

// Custom code block renderer with improved copy button
const CodeBlock = ({ node, inline, className, children, ...props }) => {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const codeString = String(children).replace(/\n$/, '');

  const handleCopy = () => {
    navigator.clipboard.writeText(codeString)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(err => console.error('Failed to copy code:', err));
  };

  return !inline ? (
    <Box sx={{ 
      position: 'relative', 
      my: 3, 
      bgcolor: 'grey.900', 
      color: 'grey.100',
      p: 3, 
      pt: 4,
      borderRadius: 2,
      overflow: 'hidden',
      boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
    }}>
      {match && (
        <Box sx={{ 
          position: 'absolute', 
          top: 0, 
          left: 0, 
          bgcolor: 'primary.main', 
          color: 'white',
          px: 2,
          py: 0.5,
          borderBottomRightRadius: 8,
          fontSize: '0.75rem',
          fontWeight: 'bold',
          textTransform: 'uppercase'
        }}>
          {match[1]}
        </Box>
      )}
      <Tooltip title={copied ? "Copied!" : "Copy code"}>
        <IconButton 
          onClick={handleCopy} 
          size="small" 
          sx={{ 
            position: 'absolute', 
            top: 8, 
            right: 8, 
            color: copied ? 'success.main' : 'grey.400',
            '&:hover': { 
              bgcolor: 'rgba(255,255,255,0.1)',
              color: 'white' 
            }
          }}
        >
          <FileCopyIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      <pre style={{ 
        margin: 0, 
        whiteSpace: 'pre-wrap', 
        wordBreak: 'break-word', 
        fontSize: '0.9rem',
        fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace'
      }}>
        <code className={className}>{children}</code>
      </pre>
    </Box>
  ) : (
    <code className={className} {...props} style={{ 
      backgroundColor: 'rgba(0,0,0,0.05)', 
      padding: '2px 4px', 
      borderRadius: '3px',
      fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace'
    }}>
      {children}
    </code>
  );
};

// Enhanced table renderer for better styling
const TableComponent = ({ node, ...props }) => (
  <Box sx={{ overflowX: 'auto', my: 3, borderRadius: 2, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
    <table style={{ 
      borderCollapse: 'collapse', 
      width: '100%', 
      minWidth: '300px', 
      backgroundColor: 'white',
      borderRadius: '8px',
      overflow: 'hidden'
    }}>
      <tbody {...props} />
    </table>
  </Box>
);

const TableCellComponent = ({ node, ...props }) => (
  <td style={{ 
    border: '1px solid #e0e0e0', 
    padding: '12px 16px', 
    textAlign: 'left',
    fontSize: '0.95rem',
    lineHeight: '1.5'
  }} {...props} />
);

const TableHeaderComponent = ({ node, ...props }) => (
  <th style={{ 
    border: '1px solid #e0e0e0', 
    padding: '14px 16px', 
    textAlign: 'left', 
    backgroundColor: '#f5f7fa', 
    fontWeight: 'bold',
    fontSize: '0.95rem'
  }} {...props} />
);

// Enhanced Markdown component with all plugins and custom renderers
const EnhancedMarkdown = ({ children }) => (
  <ReactMarkdown 
    remarkPlugins={[remarkGfm, [remarkFootnotes, {inlineNotes: true}], remarkToc]}
    rehypePlugins={[rehypeRaw, rehypeSlug, [rehypeAutolinkHeadings, {behavior: 'wrap'}]]}
    components={{ 
      img: ImgComponent, 
      code: CodeBlock, 
      table: TableComponent, 
      td: TableCellComponent, 
      th: TableHeaderComponent,
      a: ({node, ...props}) => (
        <Link 
          {...props} 
          target={props.href?.startsWith('http') ? '_blank' : undefined}
          rel={props.href?.startsWith('http') ? 'noopener noreferrer' : undefined}
          sx={{ 
            display: 'inline-flex', 
            alignItems: 'center',
            textDecoration: 'none',
            color: 'primary.main',
            '&:hover': { textDecoration: 'underline' }
          }}
        >
          {props.children}
          {props.href?.startsWith('http') && (
            <LinkIcon sx={{ ml: 0.5, fontSize: '0.9rem', opacity: 0.7 }} />
          )}
        </Link>
      )
    }}
  >
    {children}
  </ReactMarkdown>
);

// Enhanced TabPanel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`report-tabpanel-${index}`}
      aria-labelledby={`report-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 3, pb: 2, px: { xs: 0, sm: 1 } }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function ReportDisplay({ reportData }) {
  const [currentTab, setCurrentTab] = useState(0);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Extract sections from markdown based on the comprehensive report structure
  const extractSections = () => {
    if (!reportData?.report_markdown) return {};
    
    const markdown = reportData.report_markdown;
    
    // Match the comprehensive report structure
    const sections = {
      summary: markdown.match(/## 1\.\s*Executive Summary([\s\S]*?)(?=## 2\.)/i)?.[1]?.trim() || 'Executive Summary not available.',
      introduction: markdown.match(/## 2\.\s*Introduction & Background([\s\S]*?)(?=## 3\.)/i)?.[1]?.trim() || 'Introduction not available.',
      methodology: markdown.match(/## 3\.\s*Research Methodology([\s\S]*?)(?=## 4\.)/i)?.[1]?.trim() || 'Methodology not available.',
      findings: markdown.match(/## 4\.\s*Key Findings & Analysis([\s\S]*?)(?=## 5\.)/i)?.[1]?.trim() || 'Findings not available.',
      quantitative: markdown.match(/## 5\.\s*Quantitative Analysis([\s\S]*?)(?=## 6\.)/i)?.[1]?.trim() || 'Quantitative Analysis not available.',
      visualizations: markdown.match(/## 6\.\s*Visualizations & Data Representation([\s\S]*?)(?=## 7\.)/i)?.[1]?.trim() || 'Visualizations not available.',
      comparative: markdown.match(/## 7\.\s*Comparative Analysis([\s\S]*?)(?=## 8\.)/i)?.[1]?.trim() || 'Comparative Analysis not available.',
      implications: markdown.match(/## 8\.\s*Implications & Impact Assessment([\s\S]*?)(?=## 9\.)/i)?.[1]?.trim() || 'Implications not available.',
      recommendations: markdown.match(/## 9\.\s*Detailed Recommendations([\s\S]*?)(?=## 10\.)/i)?.[1]?.trim() || 'Recommendations not available.',
      conclusion: markdown.match(/## 10\.\s*Conclusion([\s\S]*?)(?=## 11\.)/i)?.[1]?.trim() || 'Conclusion not available.',
      appendices: markdown.match(/## 11\.\s*Appendices([\s\S]*?)(?=---|$)/i)?.[1]?.trim() || 'Appendices not available.',
      
      // Extract any code blocks that might be in the appendices
      code: reportData.statistical_code || reportData.generated_code || '',
    };
    
    return sections;
  };

  const reportSections = extractSections();

  // Check for mandatory fields - with our backend changes, these should always be present
  const isDataMissing = !reportSections.quantitative || reportSections.quantitative.includes('not available');
  const areChartsMissing = !reportData?.charts || reportData.charts.length === 0;
  
  // We've made backend changes to ensure these are always generated, so these warnings should never show

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const downloadReport = () => {
    const blob = new Blob([reportData.report_markdown], { type: 'text/markdown;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `research_report_${reportData.project_id}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showSnackbar('Report downloaded successfully');
  };

  const printReport = () => {
    const printWindow = window.open('', '_blank');
    
    if (!printWindow) {
      showSnackbar('Please allow pop-ups to print the report');
      return;
    }
    
    // Create a styled HTML document for printing
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>Research Report - ${reportData.project_id}</title>
          <style>
            body {
              font-family: Arial, sans-serif;
              line-height: 1.6;
              color: #333;
              max-width: 800px;
              margin: 0 auto;
              padding: 20px;
            }
            h1, h2, h3 { color: #1976d2; }
            img { max-width: 100%; height: auto; }
            table {
              border-collapse: collapse;
              width: 100%;
              margin: 20px 0;
            }
            th, td {
              border: 1px solid #ddd;
              padding: 12px;
              text-align: left;
            }
            th { background-color: #f5f5f5; }
            code {
              background-color: #f5f5f5;
              padding: 2px 4px;
              border-radius: 4px;
              font-family: monospace;
            }
            pre {
              background-color: #f5f5f5;
              padding: 15px;
              border-radius: 4px;
              overflow-x: auto;
            }
            .header {
              border-bottom: 1px solid #ddd;
              padding-bottom: 20px;
              margin-bottom: 20px;
            }
            .footer {
              border-top: 1px solid #ddd;
              padding-top: 20px;
              margin-top: 40px;
              font-size: 0.9em;
              color: #666;
            }
            @media print {
              body { font-size: 12pt; }
              h1 { font-size: 18pt; }
              h2 { font-size: 16pt; }
              h3 { font-size: 14pt; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>Research Report</h1>
            <p>Project ID: ${reportData.project_id}</p>
            <p>Generated: ${new Date().toLocaleString()}</p>
          </div>
          ${reportData.report_markdown.replace(/```/g, '')}
          <div class="footer">
            <p>Generated by AI Research Agent</p>
          </div>
          <script>
            window.onload = function() { window.print(); }
          </script>
        </body>
      </html>
    `);
    
    printWindow.document.close();
    showSnackbar('Preparing report for printing...');
  };

  const showSnackbar = (message) => {
    setSnackbarMessage(message);
    setSnackbarOpen(true);
  };

  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };

  // Handle missing data
  if (!reportData) {
    return (
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2, bgcolor: 'background.paper', textAlign: 'center' }}>
        <InsightsIcon sx={{ fontSize: 60, color: 'primary.light', mb: 2, opacity: 0.7 }} />
        <Typography variant="h5" sx={{ mb: 2, fontWeight: 'medium' }}>
          No Report Data Available
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Start a research task to generate insights and visualizations.
        </Typography>
        <Chip label="Ready to Research" color="primary" variant="outlined" />
      </Paper>
    );
  }

  if (reportData.error_message) {
    return (
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2, bgcolor: 'background.paper' }}>
        <Alert severity="error" sx={{ borderRadius: 2, mb: 3 }}>
          <Typography variant="h6">Research Error</Typography>
          <Typography>{reportData.error_message}</Typography>
        </Alert>
        <Button variant="outlined" color="primary" startIcon={<InsightsIcon />}>
          Try Again
        </Button>
      </Paper>
    );
  }

  if (!reportData.report_markdown) {
    return (
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2, bgcolor: 'background.paper', textAlign: 'center' }}>
        <DataUsageIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2, opacity: 0.8 }} />
        <Typography variant="h5" sx={{ mb: 2, fontWeight: 'medium' }}>
          Research in Progress
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Your research is being processed. Please check back shortly.
        </Typography>
        <Chip label="Processing" color="primary" />
      </Paper>
    );
  }

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: { xs: 2, sm: 3, md: 4 }, 
        borderRadius: 2, 
        bgcolor: 'background.paper',
        overflow: 'hidden'
      }}
    >
      {/* Header with actions */}
      <Box 
        sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', sm: 'row' },
          justifyContent: 'space-between', 
          alignItems: { xs: 'flex-start', sm: 'center' }, 
          mb: 3,
          pb: 2,
          borderBottom: '1px solid',
          borderColor: 'divider'
        }}
      >
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'primary.main', mb: 0.5 }}>
            Research Report
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Project ID: {reportData.project_id}
            </Typography>
            <Chip 
              label="AI Generated" 
              size="small" 
              color="primary" 
              variant="outlined" 
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', mt: { xs: 2, sm: 0 }, gap: 1 }}>
          <Tooltip title="Print Report">
            <IconButton onClick={printReport} color="primary" size={isMobile ? "small" : "medium"}>
              <PrintIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Download Markdown">
            <IconButton onClick={downloadReport} color="primary" size={isMobile ? "small" : "medium"}>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Warnings for missing mandatory fields - only show if needed */}
      {(isDataMissing || areChartsMissing) && (
        <Alert 
          severity="warning" 
          sx={{ 
            mb: 3, 
            borderRadius: 2,
            '& .MuiAlert-message': { width: '100%' }
          }}
        >
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Typography variant="subtitle2">Report Quality Issues:</Typography>
            {isDataMissing && (
              <Typography variant="body2">• Quantitative data analysis is missing or incomplete</Typography>
            )}
            {areChartsMissing && (
              <Typography variant="body2">• No charts were generated for this report</Typography>
            )}
          </Box>
        </Alert>
      )}

      {/* Tabs navigation */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs 
          value={currentTab} 
          onChange={handleTabChange} 
          aria-label="report sections" 
          variant="scrollable" 
          scrollButtons="auto"
          allowScrollButtonsMobile
          sx={{ 
            '& .MuiTab-root': { 
              textTransform: 'none', 
              fontWeight: 'medium',
              fontSize: { xs: '0.75rem', sm: '0.85rem' },
              minHeight: { xs: '48px', sm: '56px' },
              py: { xs: 1, sm: 1.5 }
            } 
          }}
        >
          <Tab icon={<SummarizeIcon fontSize="small" />} iconPosition="start" label="Summary" />
          <Tab icon={<InsightsIcon fontSize="small" />} iconPosition="start" label="Introduction" />
          <Tab label="Methodology" />
          <Tab label="Key Findings" />
          <Tab icon={<DataUsageIcon fontSize="small" />} iconPosition="start" label="Quantitative" />
          <Tab icon={<BarChartIcon fontSize="small" />} iconPosition="start" label="Visualizations" />
          <Tab label="Comparative" />
          <Tab label="Implications" />
          <Tab icon={<RecommendIcon fontSize="small" />} iconPosition="start" label="Recommendations" />
          <Tab label="Conclusion" />
          <Tab label="Appendices" />
          <Tab label="Full Report" />
        </Tabs>
      </Box>

      {/* Tab content with enhanced styling for advanced markdown */}
      <Box sx={{ 
        '& h1': { fontSize: '1.8rem', fontWeight: 'bold', mb: 2, color: 'primary.dark', mt: 1, pb: 1, borderBottom: '1px solid', borderColor: 'divider' }, 
        '& h2': { fontSize: '1.5rem', fontWeight: 'bold', mb: 2, color: 'primary.main', mt: 3, pb: 0.5 },
        '& h3': { fontSize: '1.2rem', fontWeight: 'bold', mb: 1.5, color: 'text.primary', mt: 2.5 },
        '& h4': { fontSize: '1.1rem', fontWeight: 'bold', mb: 1.5, color: 'text.secondary', mt: 2 },
        '& h5': { fontSize: '1rem', fontWeight: 'bold', mb: 1, fontStyle: 'italic' },
        '& p': { mb: 2, lineHeight: 1.7, fontSize: '1rem' },
        '& ul, & ol': { pl: 4, mb: 2, '& li': { mb: 1 } },
        '& blockquote': { 
          borderLeft: '4px solid', 
          borderColor: 'primary.main', 
          pl: 2, 
          ml: 0, 
          mb: 2, 
          bgcolor: 'grey.50',
          py: 1,
          borderRadius: '0 4px 4px 0',
          fontStyle: 'italic'
        },
        '& a': {
          color: 'primary.main',
          textDecoration: 'none',
          '&:hover': {
            textDecoration: 'underline'
          }
        },
        '& strong': {
          fontWeight: 'bold',
          color: 'text.primary'
        },
        '& hr': {
          my: 3,
          borderColor: 'divider'
        },
        '& .footnotes': {
          mt: 4,
          pt: 2,
          borderTop: '1px solid',
          borderColor: 'divider',
          fontSize: '0.9rem',
          color: 'text.secondary',
          '& ol': {
            pl: 3
          }
        },
        '& sup': {
          fontSize: '0.7rem',
          color: 'primary.main',
          ml: 0.5
        },
        '& table': {
          width: '100%',
          borderCollapse: 'collapse',
          mb: 3
        }
      }}>
        {/* Executive Summary */}
        <TabPanel value={currentTab} index={0}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <SummarizeIcon color="primary" sx={{ mr: 1.5, fontSize: '1.8rem' }} />
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Executive Summary
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.summary}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Introduction & Background */}
        <TabPanel value={currentTab} index={1}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <InsightsIcon color="primary" sx={{ mr: 1.5, fontSize: '1.8rem' }} />
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Introduction & Background
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.introduction}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Research Methodology */}
        <TabPanel value={currentTab} index={2}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Research Methodology
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.methodology}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Key Findings & Analysis */}
        <TabPanel value={currentTab} index={3}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Key Findings & Analysis
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.findings}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Quantitative Analysis */}
        <TabPanel value={currentTab} index={4}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <DataUsageIcon color="primary" sx={{ mr: 1.5, fontSize: '1.8rem' }} />
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Quantitative Analysis
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.quantitative}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Visualizations & Data Representation */}
        <TabPanel value={currentTab} index={5}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <BarChartIcon color="primary" sx={{ mr: 1.5, fontSize: '1.8rem' }} />
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Visualizations & Data Representation
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          
          {/* First show the visualization section text */}
          <EnhancedMarkdown>
            {reportSections.visualizations}
          </EnhancedMarkdown>
          
          {/* Then show the actual chart cards */}
          {reportData.charts && reportData.charts.length > 0 ? (
            <Grid container spacing={3} sx={{ mt: 2 }}>
              {reportData.charts.map((chart, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Card sx={{ 
                    borderRadius: 2, 
                    boxShadow: '0 6px 16px rgba(0,0,0,0.1)',
                    transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: '0 10px 20px rgba(0,0,0,0.15)'
                    }
                  }}>
                    <CardMedia
                      component="img"
                      image={chart.url}
                      alt={chart.title || `Chart ${index + 1}`}
                      sx={{ 
                        height: 220, 
                        objectFit: 'contain', 
                        p: 2,
                        bgcolor: 'grey.50'
                      }}
                    />
                    <CardContent sx={{ pb: 2 }}>
                      <Typography 
                        variant="subtitle1" 
                        align="center" 
                        sx={{ 
                          fontWeight: 'medium',
                          color: 'text.primary'
                        }}
                      >
                        {chart.title || `Chart ${index + 1}`}
                      </Typography>
                      {chart.description && (
                        <Typography 
                          variant="body2" 
                          align="center" 
                          color="text.secondary"
                          sx={{ mt: 1 }}
                        >
                          {chart.description}
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box sx={{ 
              p: 3, 
              bgcolor: 'grey.50', 
              borderRadius: 2, 
              textAlign: 'center',
              mt: 2
            }}>
              <BarChartIcon sx={{ fontSize: 40, color: 'text.secondary', mb: 1, opacity: 0.7 }} />
              <Typography color="text.secondary" sx={{ mb: 1 }}>
                No charts were generated for this report.
              </Typography>
              <Typography variant="body2" color="text.secondary">
                The report may still contain valuable insights in text form.
              </Typography>
            </Box>
          )}
        </TabPanel>

        {/* Comparative Analysis */}
        <TabPanel value={currentTab} index={6}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Comparative Analysis
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.comparative}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Implications & Impact Assessment */}
        <TabPanel value={currentTab} index={7}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Implications & Impact Assessment
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.implications}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Detailed Recommendations */}
        <TabPanel value={currentTab} index={8}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <RecommendIcon color="primary" sx={{ mr: 1.5, fontSize: '1.8rem' }} />
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Detailed Recommendations
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.recommendations}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Conclusion */}
        <TabPanel value={currentTab} index={9}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Conclusion
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.conclusion}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Appendices */}
        <TabPanel value={currentTab} index={10}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Appendices
            </Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportSections.appendices}
          </EnhancedMarkdown>
        </TabPanel>

        {/* Full Report */}
        <TabPanel value={currentTab} index={11}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'medium' }}>
              Complete Research Report
            </Typography>
            <Chip 
              label="Full Document" 
              size="small" 
              color="primary" 
              sx={{ ml: 2, height: 24 }}
            />
          </Box>
          <Divider sx={{ mb: 3 }} />
          <EnhancedMarkdown>
            {reportData.report_markdown}
          </EnhancedMarkdown>
        </TabPanel>
      </Box>
      
      {/* Footer */}
      <Box sx={{ 
        mt: 4, 
        pt: 2, 
        borderTop: '1px solid', 
        borderColor: 'divider',
        display: 'flex',
        flexDirection: { xs: 'column', sm: 'row' },
        justifyContent: 'space-between',
        alignItems: { xs: 'flex-start', sm: 'center' },
        gap: 1
      }}>
        <Typography variant="body2" color="text.secondary">
          Generated by AI Research Agent
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button 
            variant="outlined" 
            size="small" 
            startIcon={<DownloadIcon />}
            onClick={downloadReport}
          >
            Download
          </Button>
          <Button 
            variant="contained" 
            size="small" 
            startIcon={<PrintIcon />}
            onClick={printReport}
          >
            Print
          </Button>
        </Box>
      </Box>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={4000}
        onClose={handleCloseSnackbar}
        message={snackbarMessage}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </Paper>
  );
}

export default ReportDisplay;