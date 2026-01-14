import React, { useCallback, useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  useTheme,
  Drawer,
  Divider
} from '@mui/material';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
  Handle,
  Position
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import CloseIcon from '@mui/icons-material/Close';

const NODE_COLORS = {
  pending: '#9e9e9e',
  researching: '#2196f3',
  completed: '#4caf50',
  error: '#f44336'
};

function QuestionNode({ data, selected }) {
  const theme = useTheme();

  return (
    <Paper
      elevation={selected ? 4 : 1}
      sx={{
        p: 1.5,
        minWidth: 180,
        maxWidth: 250,
        border: selected ? `2px solid ${theme.palette.primary.main}` : 'none',
        borderLeft: `4px solid ${NODE_COLORS[data.status] || NODE_COLORS.pending}`,
        bgcolor: 'background.paper',
        transition: 'all 0.2s ease'
      }}
    >
      <Handle type="target" position={Position.Top} />

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
        <Chip
          label={data.status}
          size="small"
          sx={{
            height: 18,
            fontSize: '0.65rem',
            bgcolor: NODE_COLORS[data.status] || NODE_COLORS.pending,
            color: 'white'
          }}
        />
      </Box>

      <Typography
        variant="body2"
        sx={{
          fontWeight: 500,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical'
        }}
      >
        {data.label}
      </Typography>

      {data.sources && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
          {data.sources} sources
        </Typography>
      )}

      <Handle type="source" position={Position.Bottom} />
    </Paper>
  );
}

const nodeTypes = {
  question: QuestionNode
};

function ResearchMindmap({
  subQuestions = [],
  onNodeClick = () => {}
}) {
  const theme = useTheme();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const reactFlowWrapper = React.useRef(null);

  // Transform sub-questions to ReactFlow nodes
  useEffect(() => {
    if (subQuestions.length === 0) {
      setNodes([
        {
          id: 'main',
          type: 'question',
          position: { x: 250, y: 50 },
          data: { label: 'Start a research to see the mindmap', status: 'pending' }
        }
      ]);
      setEdges([]);
      return;
    }

    const newNodes = [];
    const newEdges = [];

    // Root node (main question)
    newNodes.push({
      id: 'main',
      type: 'question',
      position: { x: 250, y: 50 },
      data: {
        label: subQuestions[0]?.question || 'Research Question',
        status: subQuestions[0]?.status || 'pending',
        sources: subQuestions[0]?.sources?.length || 0
      }
    });

    // Child nodes (sub-questions)
    subQuestions.slice(1).forEach((sq, index) => {
      const nodeId = `sq-${sq.id || index}`;
      const x = 100 + (index % 3) * 200;
      const y = 150 + Math.floor(index / 3) * 120;

      newNodes.push({
        id: nodeId,
        type: 'question',
        position: { x, y },
        data: {
          label: sq.question,
          status: sq.status || 'pending',
          sources: sq.sources?.length || 0
        }
      });

      newEdges.push({
        id: `e-main-${nodeId}`,
        source: 'main',
        target: nodeId,
        animated: sq.status === 'researching',
        style: {
          stroke: NODE_COLORS[sq.status] || NODE_COLORS.pending,
          strokeWidth: 2
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: NODE_COLORS[sq.status] || NODE_COLORS.pending
        }
      });
    });

    setNodes(newNodes);
    setEdges(newEdges);
  }, [subQuestions, setNodes, setEdges]);

  const handleNodeClick = useCallback((event, node) => {
    const subQuestion = subQuestions.find(
      (sq, idx) => `sq-${sq.id || idx}` === node.id || node.id === 'main'
    );
    setSelectedNode(subQuestion || node.data);
    setDrawerOpen(true);
    onNodeClick(node, subQuestion);
  }, [subQuestions, onNodeClick]);

  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdge(params, eds));
  }, [setEdges]);

  const handleZoomIn = () => {
    // Zoom functionality would require reactFlowInstance
  };

  const handleZoomOut = () => {
    // Zoom functionality would require reactFlowInstance
  };

  const handleCenter = () => {
    // Center view functionality
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        position: 'relative'
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 1,
          borderBottom: `1px solid ${theme.palette.divider}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          bgcolor: 'background.paper'
        }}
      >
        <Typography variant="subtitle1" fontWeight={500}>
          Research Mind Map
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          <Tooltip title="Zoom In">
            <IconButton size="small" onClick={handleZoomIn}>
              <ZoomInIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton size="small" onClick={handleZoomOut}>
              <ZoomOutIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Center">
            <IconButton size="small" onClick={handleCenter}>
              <CenterFocusStrongIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* ReactFlow Container */}
      <Box
        ref={reactFlowWrapper}
        sx={{ flex: 1, bgcolor: theme.palette.grey[50] }}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          fitView
          attributionPosition="bottom-left"
        >
          <Background color="#aaa" gap={16} />
          <Controls />
          <MiniMap
            nodeColor={(node) => NODE_COLORS[node.data?.status] || '#ccc'}
            maskColor="rgba(0, 0, 0, 0.1)"
          />
        </ReactFlow>
      </Box>

      {/* Legend */}
      <Box
        sx={{
          p: 1,
          borderTop: `1px solid ${theme.palette.divider}`,
          display: 'flex',
          gap: 2,
          bgcolor: 'background.paper',
          justifyContent: 'center'
        }}
      >
        {Object.entries(NODE_COLORS).map(([status, color]) => (
          <Box key={status} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: color
              }}
            />
            <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
              {status}
            </Typography>
          </Box>
        ))}
      </Box>

      {/* Detail Drawer */}
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        PaperProps={{
          sx: { width: 350, p: 2 }
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Question Details</Typography>
          <IconButton onClick={() => setDrawerOpen(false)}>
            <CloseIcon />
          </IconButton>
        </Box>

        {selectedNode && (
          <>
            <Box sx={{ mb: 2 }}>
              <Chip
                label={selectedNode.status || 'unknown'}
                sx={{
                  bgcolor: NODE_COLORS[selectedNode.status] || '#ccc',
                  color: 'white',
                  mb: 1
                }}
              />
              <Typography variant="body1" fontWeight={500}>
                {selectedNode.question || selectedNode.label}
              </Typography>
            </Box>

            <Divider sx={{ my: 2 }} />

            {selectedNode.answer && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Answer
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {selectedNode.answer}
                </Typography>
              </Box>
            )}

            {selectedNode.sources && selectedNode.sources.length > 0 && (
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Sources ({selectedNode.sources.length})
                </Typography>
                {selectedNode.sources.map((source, idx) => (
                  <Chip
                    key={idx}
                    label={source.slice(0, 30) + '...'}
                    size="small"
                    variant="outlined"
                    sx={{ mr: 0.5, mb: 0.5 }}
                  />
                ))}
              </Box>
            )}
          </>
        )}
      </Drawer>
    </Box>
  );
}

export default ResearchMindmap;
