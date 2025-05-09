import React, { useState } from 'react';
import { Paper, Typography, Box, Tabs, Tab, Card, CardMedia, CardContent, Divider, IconButton, Tooltip } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import FileCopyIcon from '@mui/icons-material/FileCopy';
import DownloadIcon from '@mui/icons-material/Download';

// Custom image renderer (same as before)
const ImgComponent = ({node, ...props}) => (
  <img {...props} style={{maxWidth: '100%', height: 'auto', marginTop: '10px', marginBottom: '10px', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)'}} alt={props.alt || 'Chart'} />
);

// Custom Code block renderer for Markdown with copy button
const CodeBlock = ({node, inline, className, children, ...props}) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeString = String(children).replace(/\n$/, '');

  const handleCopy = () => {
    navigator.clipboard.writeText(codeString)
      .then(() => alert('Copied to clipboard!'))
      .catch(err => console.error('Failed to copy text: ', err));
  };

  return !inline ? (
    <Box sx={{ position: 'relative', my: 2, backgroundColor: 'grey.200', p: 1.5, borderRadius: 1, '& pre': {margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all'} }}>
      <Tooltip title="Copy code" placement="top">
        <IconButton onClick={handleCopy} size="small" sx={{ position: 'absolute', top: 4, right: 4, color: 'grey.700' }}>
          <FileCopyIcon fontSize="inherit" />
        </IconButton>
      </Tooltip>
      <pre {...props} style={{fontSize: '0.875rem'}}>
        <code className={className}>{children}</code>
      </pre>
    </Box>
  ) : (
    <code className={className} {...props}>
      {children}
    </code>
  );
};


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
        <Box sx={{ pt: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function ReportDisplay({ reportData }) {
  const [currentTab, setCurrentTab] = useState(0);

  if (!reportData || (!reportData.report_markdown && !reportData.error_message)) {
    return <Typography sx={{mt:3, textAlign:'center'}}>No report data available or research is still in progress.</Typography>;
  }
   if (reportData.error_message) {
     return (
        <Alert severity="error" sx={{mt:3}}>
            <Typography variant="h6">Research Error</Typography>
            <Typography>{reportData.error_message}</Typography>
        </Alert>
     );
   }


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
  };
  
  // Extract sections from markdown for tabs (simple split, can be more robust)
  const reportSections = {
    summary: reportData.report_markdown.match(/Section 1: Executive Summary([\s\S]*?)Section 2:/)?.[1] || "Not available.",
    plan: reportData.report_markdown.match(/Section 2: Research Plan & Methodology([\s\S]*?)Section 3:/)?.[1] || "Not available.",
    qualitative: reportData.report_markdown.match(/Section 3: Qualitative Findings & Synthesis([\s\S]*?)Section 4:/)?.[1] || "Not available.",
    quantitative: reportData.report_markdown.match(/Section 4: Quantitative Data & Statistical Analysis([\s\S]*?)Section 5:/)?.[1] || "Not available.",
    // Visualizations are handled separately below
    conclusion: reportData.report_markdown.match(/Section 6: Conclusions & Recommendations([\s\S]*)/)?.[1] || "Not available.",
  };


  return (
    <Paper elevation={3} sx={{ p: {xs: 1, sm: 2, md: 3}, mt: 0 }}>
      <Box sx={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb:2}}>
        <Typography variant="h5">Research Report</Typography>
        <Tooltip title="Download Full Markdown Report">
            <IconButton onClick={downloadReport} color="primary">
                <DownloadIcon />
            </IconButton>
        </Tooltip>
      </Box>
      <Typography variant="caption" color="textSecondary" gutterBottom>Project ID: {reportData.project_id}</Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mt: 2 }}>
        <Tabs value={currentTab} onChange={handleTabChange} aria-label="report sections tabs" variant="scrollable" scrollButtons="auto">
          <Tab label="Summary" id="report-tab-0" />
          <Tab label="Plan & Method" id="report-tab-1" />
          <Tab label="Qualitative Insights" id="report-tab-2" />
          <Tab label="Quantitative & Stats" id="report-tab-3" />
          <Tab label="Charts" id="report-tab-4" />
          <Tab label="Conclusion" id="report-tab-5" />
          <Tab label="Full Report" id="report-tab-6" />
        </Tabs>
      </Box>

      <Box className="markdown-report-content" sx={{mt: 1, fontFamily: 'inherit', '& table': {borderCollapse: 'collapse', width: 'auto', minWidth: '50%', margin: '1em 0'}, '& th, & td': {border: '1px solid #ddd', padding: '8px', textAlign: 'left'}, '& th': {backgroundColor: '#f2f2f2'} }}>
        <TabPanel value={currentTab} index={0}>
            <ReactMarkdown rehypePlugins={[rehypeRaw]} components={{img: ImgComponent, code: CodeBlock}}>{reportSections.summary}</ReactMarkdown>
        </TabPanel>
        <TabPanel value={currentTab} index={1}>
            <ReactMarkdown rehypePlugins={[rehypeRaw]} components={{img: ImgComponent, code: CodeBlock}}>{reportSections.plan}</ReactMarkdown>
        </TabPanel>
        <TabPanel value={currentTab} index={2}>
            <ReactMarkdown rehypePlugins={[rehypeRaw]} components={{img: ImgComponent, code: CodeBlock}}>{reportSections.qualitative}</ReactMarkdown>
        </TabPanel>
        <TabPanel value={currentTab} index={3}>
            <ReactMarkdown rehypePlugins={[rehypeRaw]} components={{img: ImgComponent, code: CodeBlock}}>{reportSections.quantitative}</ReactMarkdown>
        </TabPanel>
        <TabPanel value={currentTab} index={4}>
            <Typography variant="h6" gutterBottom>Generated Charts</Typography>
            {reportData.charts && reportData.charts.length > 0 ? (
            <Grid container spacing={2}>
                {reportData.charts.map((chart, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card>
                    <CardMedia
                        component="img"
                        image={chart.url}
                        alt={chart.title || `Chart ${index + 1}`}
                        sx={{ height: 200, objectFit: 'contain', p:1, borderBottom: '1px solid #eee' }}
                    />
                    <CardContent sx={{p:1.5}}>
                        <Typography variant="caption" component="div" align="center">
                        {chart.title || `Chart ${index + 1}`}
                        </Typography>
                    </CardContent>
                    </Card>
                </Grid>
                ))}
            </Grid>
            ) : (
            <Typography>No charts were generated for this report.</Typography>
            )}
        </TabPanel>
         <TabPanel value={currentTab} index={5}>
            <ReactMarkdown rehypePlugins={[rehypeRaw]} components={{img: ImgComponent, code: CodeBlock}}>{reportSections.conclusion}</ReactMarkdown>
        </TabPanel>
        <TabPanel value={currentTab} index={6}>
            <ReactMarkdown rehypePlugins={[rehypeRaw]} components={{img: ImgComponent, code: CodeBlock}}>{reportData.report_markdown}</ReactMarkdown>
        </TabPanel>
      </Box>
    </Paper>
  );
}

export default ReportDisplay;