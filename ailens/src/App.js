import React, { useState, useEffect, useRef } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  AppBar, Toolbar, Typography, Container, Paper, 
  Button, IconButton, CircularProgress, MenuItem,
  Select, FormControl, InputLabel, LinearProgress,
  Dialog, DialogTitle, DialogContent, DialogContentText,
  Snackbar, Alert, Box, Grid
} from '@mui/material';
import {
  CameraAlt as CameraIcon,
  FlipCameraAndroid as FlipCameraIcon,
  PhotoLibrary as GalleryIcon,
  Send as SendIcon,
  Memory as MemoryIcon,
  EjectOutlined as EjectIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { io } from 'socket.io-client';
import './App.css';

// Theme configuration
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
});

function App() {
  // State management
  const [cameraStream, setCameraStream] = useState(null);
  const [facingMode, setFacingMode] = useState('environment'); // 'environment' for rear, 'user' for front
  const [capturedImage, setCapturedImage] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [connectionDialogOpen, setConnectionDialogOpen] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [recognitionResult, setRecognitionResult] = useState(null);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  const [availableCameras, setAvailableCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [processingId, setProcessingId] = useState(null);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState(null);
  const [progressInterval, setProgressIntervalState] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const processingStartTimeRef = useRef(null);

  // Dropzone configuration for gallery uploads
  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif']
    },
    onDrop: (acceptedFiles) => {
      handleGalleryUpload(acceptedFiles[0]);
    }
  });

  // Initialize socket.io connection
  useEffect(() => {
    // Check if we're on HTTPS
    if (window.location.protocol !== 'https:') {
      showNotification('This app works best with HTTPS to access camera features.', 'warning');
    }

    // Test server reachability
    fetch('/api/models')
      .then(response => {
        console.log('Server is reachable!', response);
        if (!response.ok) {
          throw new Error('Server returned an error response');
        }
        return response.json();
      })
      .then(data => {
        console.log('Models available:', data);
        if (data.models && data.models.length > 0) {
          setAvailableModels(data.models);
          setSelectedModel(data.models[0]);
        }
      })
      .catch(error => {
        console.error('Error reaching server:', error);
        showNotification('Error connecting to server. Please check if the backend is running.', 'error');
      });

    // Establish Socket.IO connection
    const socket = io('/', {
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 60000,
      transports: ['websocket', 'polling']  // Try websocket first
    });

    socket.io.on("reconnect_attempt", (attempt) => {
      console.log(`Reconnection attempt ${attempt}`);
      showNotification(`Reconnection attempt ${attempt}...`, 'warning');
    });
    
    socket.io.on("reconnect_error", (error) => {
      console.error("Reconnection error:", error);
      showNotification('Reconnection error', 'error');
    });
    
    socket.io.on("reconnect", (attempt) => {
      console.log(`Reconnected after ${attempt} attempts`);
      showNotification('Reconnected to server', 'success');
    });
    
    socketRef.current = socket;
    
    socket.on('connect', () => {
      setIsConnected(true);
      setConnectionDialogOpen(true);
      fetchAvailableModels();
    });
    
    socket.on('disconnect', () => {
      setIsConnected(false);
      showNotification('Connection to server lost. Please refresh.', 'error');
    });
    
    socket.on('models', (data) => {
      setAvailableModels(data.models);
      if (data.models.length > 0) {
        setSelectedModel(data.models[0]);
      }
    });
    
    socket.on('model_status', (data) => {
      setModelLoaded(data.loaded);
      setLoadingProgress(data.progress || 0);
      if (data.loaded) {
        showNotification(`Model ${data.name} loaded successfully`, 'success');
      }
    });
    
    socket.on('processing', (data) => {
      const progress = data.progress || 0;
      setProcessingProgress(progress);
      
      // Update processing ID if provided
      if (data.processing_id) {
        setProcessingId(data.processing_id);
      }
      
      // Calculate estimated time remaining
      if (processingStartTimeRef.current && progress > 0 && progress < 100) {
        const elapsedTime = (Date.now() - processingStartTimeRef.current) / 1000; // in seconds
        const estimatedTotal = elapsedTime / (progress / 100);
        const remaining = Math.round(estimatedTotal - elapsedTime);
        setEstimatedTimeRemaining(remaining);
      }
    });
    
    socket.on('result', (data) => {
      setRecognitionResult(data.result);
      setIsProcessing(false);
      setProcessingId(null);
      setEstimatedTimeRemaining(null);
      showNotification('Image recognition complete!', 'success');
    });
    
    socket.on('error', (data) => {
      showNotification(data.message, 'error');
      setIsProcessing(false);
      setProcessingId(null);
      setEstimatedTimeRemaining(null);
    });
    
    socket.on('cancelled', (data) => {
      showNotification('Image processing cancelled', 'info');
      setIsProcessing(false);
      setProcessingProgress(0);
      setProcessingId(null);
      setEstimatedTimeRemaining(null);
    });
    
    // Cleanup function
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      stopCamera();
    };
  }, []);

  // Fetch available cameras
  useEffect(() => {
    const getCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(device => device.kind === 'videoinput');
        setAvailableCameras(cameras);
        if (cameras.length > 0) {
          setSelectedCamera(cameras[0].deviceId);
        }
      } catch (error) {
        console.error('Error getting cameras:', error);
        showNotification('Could not access cameras', 'error');
      }
    };
    
    getCameras();
  }, []);

  // Start camera stream when selected camera changes
  useEffect(() => {
    if (selectedCamera) {
      startCamera();
    }
  }, [selectedCamera, facingMode]);

  const fetchAvailableModels = () => {
    if (socketRef.current && socketRef.current.connected) {
      socketRef.current.emit('get_models');
    } else {
      // Fallback to REST API if socket isn't connected
      fetch('/api/models')
        .then(response => response.json())
        .then(data => {
          if (data.models) {
            setAvailableModels(data.models);
            if (data.models.length > 0) {
              setSelectedModel(data.models[0]);
            }
          }
        })
        .catch(error => {
          console.error('Error fetching models:', error);
          showNotification('Error fetching available models', 'error');
        });
    }
  };

  const startCamera = async () => {
    try {
      if (cameraStream) {
        stopCamera();
      }
      
      const constraints = {
        video: selectedCamera 
          ? { deviceId: { exact: selectedCamera } } 
          : { facingMode },
        audio: false
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setCameraStream(stream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
    } catch (error) {
      console.error('Error starting camera:', error);
      showNotification('Could not access camera', 'error');
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
  };

  const switchCamera = () => {
    const newFacingMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(newFacingMode);
  };

  const captureImage = () => {
    if (!videoRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to data URL
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
    setCapturedImage(imageDataUrl);
    
    showNotification('Image captured! Ready to process.', 'info');
  };

  const handleGalleryUpload = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setCapturedImage(e.target.result);
      showNotification('Image loaded from gallery!', 'info');
    };
    reader.readAsDataURL(file);
  };

  const processImage = () => {
    if (!capturedImage || !modelLoaded) {
      showNotification('Please capture an image and ensure model is loaded', 'warning');
      return;
    }
    
    setIsProcessing(true);
    setProcessingProgress(0);
    setRecognitionResult(null);
    setEstimatedTimeRemaining(null);
    processingStartTimeRef.current = Date.now();
    
    // Set up progress simulation
    let progress = 10;
    if (progressInterval) {
      clearInterval(progressInterval);
    }
    const interval = setInterval(() => {
      progress += 5;
      if (progress >= 90) {
        clearInterval(interval);
      }
      setProcessingProgress(progress);
    }, 500);
    setProgressIntervalState(interval);
    
    // Use REST API instead of Socket.IO for processing
    fetch('/api/process-image-v2', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        image: capturedImage,
        model: selectedModel
      }),
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      setProcessingProgress(100); // Done
      return response.json();
    })
    .then(data => {
      if (data.success) {
        setRecognitionResult(data.result);
        showNotification('Image recognition complete!', 'success');
      } else {
        showNotification(data.message || 'Processing failed', 'error');
      }
    })
    .catch(error => {
      console.error('Error processing image:', error);
      showNotification(`Error: ${error.message}`, 'error');
    })
    .finally(() => {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      setProcessingProgress(100);
      setTimeout(() => {
        setIsProcessing(false);
        setProcessingId(null);
        setEstimatedTimeRemaining(null);
      }, 500); // Give a moment to show 100%
    });
  };
  
  const cancelProcessing = () => {
    if (!isProcessing) {
      return;
    }
    
    // Clear the progress simulation interval
    if (progressInterval) {
      clearInterval(progressInterval);
    }
    
    // If we have a processing ID, try to cancel via Socket.IO
    if (processingId && socketRef.current && socketRef.current.connected) {
      socketRef.current.emit('cancel_processing', {
        processing_id: processingId
      });
    } else {
      // Otherwise just reset the UI
      setIsProcessing(false);
      setProcessingProgress(0);
      setProcessingId(null);
      setEstimatedTimeRemaining(null);
      showNotification('Processing cancelled', 'info');
    }
  };

  const loadModel = () => {
    if (!selectedModel) {
      showNotification('Please select a model first', 'warning');
      return;
    }
    
    setLoadingProgress(0);
    
    if (socketRef.current && socketRef.current.connected) {
      socketRef.current.emit('load_model', {
        model: selectedModel
      });
    } else {
      // Fallback to REST API if socket isn't connected
      fetch('/api/model/load', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel
        }),
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          // Since we can't get progress updates without socket, show a notification
          showNotification(`Started loading model ${selectedModel}`, 'info');
          // We'll need to poll for model status
          checkModelLoadingStatus();
        } else {
          showNotification(data.message || 'Failed to load model', 'error');
        }
      })
      .catch(error => {
        console.error('Error loading model:', error);
        showNotification(`Error loading model: ${error.message}`, 'error');
      });
    }
  };

  const checkModelLoadingStatus = () => {
    // Poll the server every 2 seconds to check model loading status
    const checkInterval = setInterval(() => {
      fetch('/api/models')
        .then(response => response.json())
        .then(data => {
          // If we got a valid response, assume the model is loaded
          setModelLoaded(true);
          setLoadingProgress(100);
          showNotification(`Model ${selectedModel} loaded successfully`, 'success');
          clearInterval(checkInterval);
        })
        .catch(error => {
          console.error('Error checking model status:', error);
          // Keep trying
        });
    }, 2000);
    
    // Stop checking after 60 seconds
    setTimeout(() => {
      clearInterval(checkInterval);
    }, 60000);
  };

  const unloadModel = () => {
    if (!modelLoaded) {
      showNotification('No model currently loaded', 'warning');
      return;
    }
    
    if (socketRef.current && socketRef.current.connected) {
      socketRef.current.emit('unload_model');
      setModelLoaded(false);
      showNotification('Model unloaded from GPU', 'info');
    } else {
      // Fallback to REST API
      fetch('/api/model/unload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          setModelLoaded(false);
          showNotification('Model unloaded from GPU', 'success');
        } else {
          showNotification(data.message || 'Failed to unload model', 'error');
        }
      })
      .catch(error => {
        console.error('Error unloading model:', error);
        showNotification(`Error unloading model: ${error.message}`, 'error');
      });
    }
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleCameraChange = (event) => {
    setSelectedCamera(event.target.value);
  };

  const showNotification = (message, severity) => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  const closeNotification = () => {
    setNotification({
      ...notification,
      open: false
    });
  };

  // Format the estimated time remaining
  const formatTimeRemaining = (seconds) => {
    if (!seconds || isNaN(seconds)) return '';
    
    if (seconds < 60) {
      return `${seconds} sec remaining`;
    } else {
      const min = Math.floor(seconds / 60);
      const sec = seconds % 60;
      return `${min}:${sec.toString().padStart(2, '0')} remaining`;
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, height: '100vh', display: 'flex', flexDirection: 'column' }}>
        <AppBar position="static" color="primary">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              AI Vision Recognition
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Box sx={{ 
                width: 12, 
                height: 12, 
                borderRadius: '50%', 
                bgcolor: isConnected ? 'success.main' : 'error.main',
                mr: 1 
              }} />
              <Typography variant="body2">
                {isConnected ? 'Connected' : 'Disconnected'}
              </Typography>
            </Box>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ flexGrow: 1, py: 2, display: 'flex', flexDirection: 'column' }}>
          <Grid container spacing={2} sx={{ flexGrow: 1 }}>
            {/* Camera & Image Display Section */}
            <Grid item xs={12} md={7} sx={{ height: { xs: '40vh', md: '80vh' } }}>
              <Paper 
                elevation={3} 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  overflow: 'hidden'
                }}
              >
                <Box sx={{ p: 1, bgcolor: 'background.paper', borderBottom: 1, borderColor: 'divider' }}>
                  <Grid container spacing={1} alignItems="center">
                    <Grid item>
                      <Typography variant="subtitle1">Camera View</Typography>
                    </Grid>
                    <Grid item xs>
                      <FormControl variant="outlined" size="small" fullWidth>
                        <InputLabel id="camera-select-label">Camera</InputLabel>
                        <Select
                          labelId="camera-select-label"
                          id="camera-select"
                          value={selectedCamera}
                          onChange={handleCameraChange}
                          label="Camera"
                          disabled={availableCameras.length === 0}
                        >
                          {availableCameras.map((camera) => (
                            <MenuItem key={camera.deviceId} value={camera.deviceId}>
                              {camera.label || `Camera ${camera.deviceId.substring(0, 4)}`}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item>
                      <IconButton onClick={switchCamera} color="primary" disabled={!cameraStream}>
                        <FlipCameraIcon />
                      </IconButton>
                    </Grid>
                  </Grid>
                </Box>

                <Box sx={{ position: 'relative', flexGrow: 1, bgcolor: 'black' }}>
                  {capturedImage ? (
                    <img 
                      src={capturedImage} 
                      alt="Captured" 
                      style={{ 
                        width: '100%', 
                        height: '100%', 
                        objectFit: 'contain'
                      }} 
                    />
                  ) : (
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain'
                      }}
                    />
                  )}
                  
                  {/* Hidden canvas for capturing */}
                  <canvas ref={canvasRef} style={{ display: 'none' }} />
                </Box>

                <Box sx={{ p: 1, display: 'flex', justifyContent: 'center', gap: 1 }}>
                  {!capturedImage ? (
                    <>
                      <Button
                        variant="contained"
                        color="primary"
                        startIcon={<CameraIcon />}
                        onClick={captureImage}
                        disabled={!cameraStream}
                      >
                        Capture
                      </Button>
                      <div {...getRootProps()}>
                        <input {...getInputProps()} />
                        <Button 
                          variant="outlined" 
                          color="secondary"
                          startIcon={<GalleryIcon />}
                        >
                          Gallery
                        </Button>
                      </div>
                    </>
                  ) : (
                    <>
                      <Button
                        variant="outlined"
                        color="primary"
                        onClick={() => setCapturedImage(null)}
                      >
                        Reset
                      </Button>
                      {!isProcessing ? (
                        <Button
                          variant="contained"
                          color="success"
                          startIcon={<SendIcon />}
                          onClick={processImage}
                          disabled={!modelLoaded}
                        >
                          Analyze
                        </Button>
                      ) : (
                        <Button
                          variant="contained"
                          color="error"
                          startIcon={<CancelIcon />}
                          onClick={cancelProcessing}
                        >
                          Cancel
                        </Button>
                      )}
                    </>
                  )}
                </Box>
              </Paper>
            </Grid>

            {/* Controls & Results Section */}
            <Grid item xs={12} md={5} sx={{ display: 'flex', flexDirection: 'column', height: { xs: 'auto', md: '80vh' } }}>
              {/* Model Section */}
              <Paper elevation={3} sx={{ mb: 2, p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Model Controls
                </Typography>
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel id="model-select-label">Vision Model</InputLabel>
                  <Select
                    labelId="model-select-label"
                    id="model-select"
                    value={selectedModel}
                    onChange={handleModelChange}
                    label="Vision Model"
                    disabled={modelLoaded}
                  >
                    {availableModels.map((model) => (
                      <MenuItem key={model} value={model}>
                        {model}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<MemoryIcon />}
                    onClick={loadModel}
                    disabled={modelLoaded || !selectedModel || loadingProgress > 0}
                  >
                    Load Model
                  </Button>
                  
                  <Button
                    variant="outlined"
                    color="secondary"
                    startIcon={<EjectIcon />}
                    onClick={unloadModel}
                    disabled={!modelLoaded}
                  >
                    Unload
                  </Button>
                </Box>
                
                {loadingProgress > 0 && loadingProgress < 100 && (
                  <Box sx={{ width: '100%', mt: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Loading model: {loadingProgress}%
                    </Typography>
                    <LinearProgress variant="determinate" value={loadingProgress} />
                  </Box>
                )}
                
                <Box sx={{ mt: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Status: {modelLoaded 
                      ? `${selectedModel} loaded in GPU` 
                      : 'No model loaded'
                    }
                  </Typography>
                </Box>
              </Paper>

              {/* Results Section */}
              <Paper elevation={3} sx={{ p: 2, flexGrow: 1, overflow: 'auto' }}>
                <Typography variant="h6" gutterBottom>
                  Recognition Results
                </Typography>
                
                {isProcessing && (
                  <Box sx={{ width: '100%', textAlign: 'center', py: 4 }}>
                    <CircularProgress size={60} />
                    <Typography variant="body1" sx={{ mt: 2 }}>
                      Processing image...
                    </Typography>
                    {processingProgress > 0 && (
                      <Box sx={{ width: '100%', mt: 2 }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={processingProgress} 
                          color={processingProgress < 30 ? "secondary" : processingProgress < 70 ? "primary" : "success"}
                        />
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                          <Typography variant="body2" color="text.secondary">
                            {processingProgress}% complete
                          </Typography>
                          {estimatedTimeRemaining && (
                            <Typography variant="body2" color="text.secondary">
                              {formatTimeRemaining(estimatedTimeRemaining)}
                            </Typography>
                          )}
                        </Box>
                      </Box>
                    )}
                  </Box>
                )}
                
                {!isProcessing && recognitionResult && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Detected:
                    </Typography>
                    <Typography variant="body1" component="pre" sx={{ 
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      bgcolor: 'background.paper',
                      p: 2,
                      borderRadius: 1,
                      border: 1,
                      borderColor: 'divider'
                    }}>
                      {recognitionResult}
                    </Typography>
                  </Box>
                )}
                
                {!isProcessing && !recognitionResult && (
                  <Box sx={{ 
                    width: '100%', 
                    height: '100%', 
                    display: 'flex', 
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '200px'
                  }}>
                    <Typography variant="body1" color="text.secondary">
                      Capture and analyze an image to see results
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Connection Dialog */}
      <Dialog
        open={connectionDialogOpen}
        onClose={() => setConnectionDialogOpen(false)}
      >
        <DialogTitle>Connected to Local Server</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Successfully connected to the local vision AI server. 
            You can now select a model to load and start recognizing images.
          </DialogContentText>
          <Button 
            variant="contained" 
            color="primary"
            onClick={() => setConnectionDialogOpen(false)}
            sx={{ mt: 2 }}
          >
            Got it
          </Button>
        </DialogContent>
      </Dialog>

      {/* Notification Snackbar */}
      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={closeNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={closeNotification} severity={notification.severity}>
          {notification.message}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;