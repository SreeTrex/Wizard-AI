import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data'; 
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/io.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:uuid/uuid.dart';

// Message class to store chat messages
class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final String? imageUrl; // Optional path to image

  ChatMessage({
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.imageUrl,
  });
}

void main() async {
  // Ensure Flutter is initialized
  WidgetsFlutterBinding.ensureInitialized();

  // Get available cameras with error handling
  List<CameraDescription> cameras = [];
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error getting cameras: $e');
  }

  runApp(
    MaterialApp(
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      themeMode: ThemeMode.system,
      home: cameras.isNotEmpty 
          ? VisionAnalyzerApp(camera: cameras.first)
          : ErrorScreen('No camera found or camera access denied'),
    ),
  );
}

// Error screen widget shown when camera cannot be accessed
class ErrorScreen extends StatelessWidget {
  final String errorMessage;
  
  const ErrorScreen(this.errorMessage, {Key? key}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Vision Analyzer')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error_outline, size: 80, color: Colors.red),
              SizedBox(height: 24),
              Text(
                'Camera Error',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 16),
              Text(
                errorMessage,
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16),
              ),
              SizedBox(height: 24),
              Text(
                'Please check your device permissions and ensure camera access is granted to this app.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16),
              ),
              SizedBox(height: 24),
              ElevatedButton(
                onPressed: () {
                  // Attempt to restart the app
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (context) => RestartWidget()),
                  );
                },
                child: Text('Retry Camera Access'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// Widget to attempt restarting the app and reinitializing camera
class RestartWidget extends StatefulWidget {
  @override
  _RestartWidgetState createState() => _RestartWidgetState();
}

class _RestartWidgetState extends State<RestartWidget> {
  @override
  void initState() {
    super.initState();
    _restartApp();
  }
  
  Future<void> _restartApp() async {
    // Slight delay to allow widget to build
    await Future.delayed(Duration(milliseconds: 100));
    
    // Get available cameras with error handling
    List<CameraDescription> cameras = [];
    String errorMessage = 'Could not access camera. Please check your device settings.';
    
    try {
      cameras = await availableCameras();
    } catch (e) {
      errorMessage = 'Error accessing camera: $e';
      print(errorMessage);
    }
    
    // Navigate to proper screen based on camera availability
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) => cameras.isNotEmpty 
            ? VisionAnalyzerApp(camera: cameras.first)
            : ErrorScreen(errorMessage),
      ),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: CircularProgressIndicator(),
      ),
    );
  }
}

class VisionAnalyzerApp extends StatefulWidget {
  final CameraDescription camera;

  const VisionAnalyzerApp({
    Key? key,
    required this.camera,
  }) : super(key: key);

  @override
  _VisionAnalyzerAppState createState() => _VisionAnalyzerAppState();
}

class _VisionAnalyzerAppState extends State<VisionAnalyzerApp> with WidgetsBindingObserver {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool _isProcessing = false;
  bool _cameraInitialized = false;
  String _cameraErrorMessage = '';
  String _serverUrl = 'https://db51-35-227-69-77.ngrok-free.app'; 
  TextEditingController _questionController = TextEditingController();

  // Message list for the chat interface
  List<ChatMessage> _messages = [];
  ScrollController _scrollController = ScrollController();

  // WebSocket variables for query/response channel
  WebSocketChannel? _queryResponseChannel;
  bool _isQueryResponseConnected = false;
  Timer? _queryResponsePingTimer;
  Timer? _queryResponseReconnectTimer;

  // WebSocket variables for frames channel
  WebSocketChannel? _framesChannel;
  bool _isFramesConnected = false;
  Timer? _framesPingTimer;
  Timer? _framesReconnectTimer;

  // Common WebSocket variables
  String _sessionId = const Uuid().v4(); // Generate session ID at startup
  String _wsBaseUrl = 'wss://db51-35-227-69-77.ngrok-free.app';

  // Frame capture timer
  Timer? _frameCaptureTimer;
  bool _isCapturingFrames = false;

  // Speech to text
  stt.SpeechToText _speech = stt.SpeechToText();
  bool _isListening = false;
  
  // Text to speech
  FlutterTts _flutterTts = FlutterTts();
  bool _isSpeaking = false;

  // Connectivity monitoring
  StreamSubscription? _connectivitySubscription;

  // User typing detection
  bool _isTyping = false;
  Timer? _typingTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Initialize the camera controller with robust error handling
    _initializeCamera();
    
    // Set default question
    _questionController.text = "Analyze this image and describe what you see:";
    
    // Setup typing detection
    _questionController.addListener(_onTypingDetected);
    
    // Initialize WebSocket connections
    _initWebSockets();

    // Initialize speech to text
    _initSpeech();
    
    // Initialize text to speech
    _initTts();

    // Initialize connectivity monitoring
    _monitorConnectivity();
  }

  void _onTypingDetected() {
    if (_questionController.text.isNotEmpty) {
      setState(() {
        _isTyping = true;
        _startPeriodicFrameCapture();
      });
      
      // Reset typing timer
      _typingTimer?.cancel();
      _typingTimer = Timer(Duration(seconds: 2), () {
        setState(() {
          _isTyping = false;
        });
      });
      
      // Start frame capture if not already capturing
      if (!_isCapturingFrames) {
        _startPeriodicFrameCapture();
      }
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Handle app lifecycle changes
    if (state == AppLifecycleState.resumed) {
      // App is in the foreground
      if (!_isQueryResponseConnected || !_isFramesConnected) {
        _initWebSockets(); // Reconnect when app comes to foreground
      }
      if (_isCapturingFrames) {
        _startPeriodicFrameCapture(); // Restart frame capture if it was active
      }
    } else if (state == AppLifecycleState.paused) {
      // App is in the background
      _disconnectWebSockets(); // Disconnect when app goes to background
      _stopPeriodicFrameCapture(); // Stop frame capture
    }
  }
  
  // Monitor device connectivity changes
  void _monitorConnectivity() {
    _connectivitySubscription = Connectivity().onConnectivityChanged.listen((result) {
      if (result != ConnectivityResult.none) {
        // We have connectivity, try to connect WebSockets if not connected
        if (!_isQueryResponseConnected || !_isFramesConnected) {
          _initWebSockets();
        }
      } else {
        // No connectivity
        setState(() {
          _isQueryResponseConnected = false;
          _isFramesConnected = false;
        });
      }
    });
  }
  
  // Initialize camera with better error handling
  Future<void> _initializeCamera() async {
    try {
      // Initialize with lower resolution first to improve compatibility
      _controller = CameraController(
        widget.camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      
      _initializeControllerFuture = _controller.initialize().then((_) {
        if (mounted) {
          setState(() {
            _cameraInitialized = true;
          });
        }
      }).catchError((error) {
        if (mounted) {
          setState(() {
            _cameraErrorMessage = 'Failed to initialize camera: $error';
            print(_cameraErrorMessage);
          });
        }
        return null;
      });
    } catch (e) {
      setState(() {
        _cameraErrorMessage = 'Error creating camera controller: $e';
        print(_cameraErrorMessage);
      });
    }
  }

  // Initialize both WebSocket connections
  void _initWebSockets() {
    _initQueryResponseWebSocket();
    _initFramesWebSocket();
  }

  // Initialize WebSocket connection for query/response
  void _initQueryResponseWebSocket() {
    try {
      // Close existing connection if any
      _disconnectQueryResponseWebSocket();
      
      // Create a full WebSocket URL with session ID
      final wsUrl = '$_wsBaseUrl/queryResponse/${_sessionId}';
      print('Connecting to Query/Response WebSocket: $wsUrl');
      
      // Create new WebSocket connection
      _queryResponseChannel = IOWebSocketChannel.connect(
        Uri.parse(wsUrl),
        pingInterval: Duration(seconds: 30),
      );
      
      // Setup message listener
      _queryResponseChannel!.stream.listen(
        (message) {
          // Process incoming message
          _handleQueryResponseMessage(message);
        },
        onDone: () {
          print('Query/Response WebSocket connection closed');
          setState(() {
            _isQueryResponseConnected = false;
          });
          // Try to reconnect
          _scheduleQueryResponseReconnect();
        },
        onError: (error) {
          print('Query/Response WebSocket error: $error');
          setState(() {
            _isQueryResponseConnected = false;
          });
          // Try to reconnect
          _scheduleQueryResponseReconnect();
        },
        cancelOnError: true,
      );
      
      // Send initial connection message
      _sendQueryResponseConnectionMessage();
      
      // Setup ping timer to keep connection alive
      _setupQueryResponsePingTimer();
      
      setState(() {
        _isQueryResponseConnected = true;
      });
    } catch (e) {
      print('Error connecting to Query/Response WebSocket: $e');
      setState(() {
        _isQueryResponseConnected = false;
      });
      // Try to reconnect
      _scheduleQueryResponseReconnect();
    }
  }

  // Initialize WebSocket connection for frames
  void _initFramesWebSocket() {
    try {
      // Close existing connection if any
      _disconnectFramesWebSocket();
      
      // Create a full WebSocket URL with session ID
      final wsUrl = '$_wsBaseUrl/frames/${_sessionId}';
      print('Connecting to Frames WebSocket: $wsUrl');
      
      // Create new WebSocket connection
      _framesChannel = IOWebSocketChannel.connect(
        Uri.parse(wsUrl),
        pingInterval: Duration(seconds: 30),
      );
      
      // Setup message listener
      _framesChannel!.stream.listen(
        (message) {
          // Process incoming message from frames channel
          _handleFramesMessage(message);
        },
        onDone: () {
          print('Frames WebSocket connection closed');
          setState(() {
            _isFramesConnected = false;
          });
          // Try to reconnect
          _scheduleFramesReconnect();
        },
        onError: (error) {
          print('Frames WebSocket error: $error');
          setState(() {
            _isFramesConnected = false;
          });
          // Try to reconnect
          _scheduleFramesReconnect();
        },
        cancelOnError: true,
      );
      
      // Send initial connection message
      _sendFramesConnectionMessage();
      
      // Setup ping timer to keep connection alive
      _setupFramesPingTimer();
      
      setState(() {
        _isFramesConnected = true;
      });
    } catch (e) {
      print('Error connecting to Frames WebSocket: $e');
      setState(() {
        _isFramesConnected = false;
      });
      // Try to reconnect
      _scheduleFramesReconnect();
    }
  }
  
  // Send initial connection message for query/response channel
  void _sendQueryResponseConnectionMessage() {
    if (_queryResponseChannel != null && _isQueryResponseConnected) {
      _queryResponseChannel!.sink.add(jsonEncode({
        'type': 'connect',
        'session_id': _sessionId,
      }));
    }
  }
  
  // Send initial connection message for frames channel
  void _sendFramesConnectionMessage() {
    if (_framesChannel != null && _isFramesConnected) {
      _framesChannel!.sink.add(jsonEncode({
        'type': 'connect',
        'session_id': _sessionId,
      }));
    }
  }
  
  // Setup ping timer for query/response channel
  void _setupQueryResponsePingTimer() {
    // Cancel existing timer if any
    _queryResponsePingTimer?.cancel();
    
    // Setup new timer
    _queryResponsePingTimer = Timer.periodic(Duration(seconds: 20), (timer) {
      if (_queryResponseChannel != null && _isQueryResponseConnected) {
        _queryResponseChannel!.sink.add(jsonEncode({
          'type': 'ping',
          'timestamp': DateTime.now().toIso8601String(),
        }));
      }
    });
  }
  
  // Setup ping timer for frames channel
  void _setupFramesPingTimer() {
    // Cancel existing timer if any
    _framesPingTimer?.cancel();
    
    // Setup new timer
    _framesPingTimer = Timer.periodic(Duration(seconds: 20), (timer) {
      if (_framesChannel != null && _isFramesConnected) {
        _framesChannel!.sink.add(jsonEncode({
          'type': 'ping',
          'timestamp': DateTime.now().toIso8601String(),
        }));
      }
    });
  }
  
  // Schedule reconnection attempt for query/response channel
  void _scheduleQueryResponseReconnect() {
    // Cancel existing timer if any
    _queryResponseReconnectTimer?.cancel();
    
    // Setup new timer with exponential backoff
    _queryResponseReconnectTimer = Timer(Duration(seconds: 5), () {
      if (!_isQueryResponseConnected) {
        _initQueryResponseWebSocket();
      }
    });
  }
  
  // Schedule reconnection attempt for frames channel
  void _scheduleFramesReconnect() {
    // Cancel existing timer if any
    _framesReconnectTimer?.cancel();
    
    // Setup new timer with exponential backoff
    _framesReconnectTimer = Timer(Duration(seconds: 5), () {
      if (!_isFramesConnected) {
        _initFramesWebSocket();
      }
    });
  }
  
  // Disconnect query/response WebSocket
  void _disconnectQueryResponseWebSocket() {
    _queryResponsePingTimer?.cancel();
    _queryResponseReconnectTimer?.cancel();
    
    if (_queryResponseChannel != null) {
      _queryResponseChannel!.sink.close();
      _queryResponseChannel = null;
    }
    
    setState(() {
      _isQueryResponseConnected = false;
    });
  }
  
  // Disconnect frames WebSocket
  void _disconnectFramesWebSocket() {
    _framesPingTimer?.cancel();
    _framesReconnectTimer?.cancel();
    
    if (_framesChannel != null) {
      _framesChannel!.sink.close();
      _framesChannel = null;
    }
    
    setState(() {
      _isFramesConnected = false;
    });
  }
  
  // Disconnect all WebSockets
  void _disconnectWebSockets() {
    _disconnectQueryResponseWebSocket();
    _disconnectFramesWebSocket();
  }
  
  // Handle incoming WebSocket messages from query/response channel
  void _handleQueryResponseMessage(dynamic message) {
    try {
      final data = jsonDecode(message);
      final messageType = data['type'];
      
      switch (messageType) {
        case 'connection_established':
          print('Query/Response WebSocket connection established');
          break;
          
        case 'processing_status':
          // Handle processing status updates
          if (data['status'] == 'processing') {
            setState(() {
              _isProcessing = true;
            });
          }
          break;
          
        case 'analysis_result':
          // Handle analysis results
          setState(() {
            _isProcessing = false;
            if (data['status'] == 'success') {
              // Add AI response to messages
              _addMessage(
                data['analysis'], 
                false, // AI message (not user)
              );
              
              // Auto-speak the response
              _speak(data['analysis']);
            } else {
              // Add error message
              _addMessage(
                'Error: ${data["error"] ?? "Unknown error"}',
                false, // AI message (not user)
              );
            }
          });
          break;
          
        case 'pong':
          // Handle pong message (ping response)
          print('Received pong from query/response server');
          break;
          
        default:
          print('Unknown message type from query/response: $messageType');
      }
    } catch (e) {
      print('Error processing Query/Response WebSocket message: $e');
    }
  }
  
  // Handle incoming WebSocket messages from frames channel
  void _handleFramesMessage(dynamic message) {
    try {
      final data = jsonDecode(message);
      final messageType = data['type'];
      
      switch (messageType) {
        case 'connection_established':
          print('Frames WebSocket connection established');
          break;
          
        case 'frame_received':
          // Confirm frame was received
          print('Frame received confirmation: ${data['frame_id']}');
          break;
          
        case 'pong':
          // Handle pong message (ping response)
          print('Received pong from frames server');
          break;
          
        default:
          print('Unknown message type from frames: $messageType');
      }
    } catch (e) {
      print('Error processing Frames WebSocket message: $e');
    }
  }
  
  // Start periodic frame capture
  void _startPeriodicFrameCapture() {
    // Cancel existing timer if any
    _stopPeriodicFrameCapture();
    
    // Set capturing flag
    _isCapturingFrames = true;
    
    // Capture a frame immediately
    _captureAndSendFrame();
    
    // Setup new timer for periodic capture
    _frameCaptureTimer = Timer.periodic(Duration(seconds: 2), (timer) {
      if (_cameraInitialized && (_isListening || _isTyping)) {
        _captureAndSendFrame();
      } else {
        // Stop capturing if not listening or typing
        _stopPeriodicFrameCapture();
      }
    });
  }
  
  // Stop periodic frame capture
  void _stopPeriodicFrameCapture() {
    _frameCaptureTimer?.cancel();
    _frameCaptureTimer = null;
    _isCapturingFrames = false;
  }
  
  // Capture and send a single frame
  Future<void> _captureAndSendFrame() async {
    if (!_cameraInitialized || !_isFramesConnected) return;
    
    try {
      // Ensure the camera is initialized
      await _initializeControllerFuture;
      
      // Capture a photo
      XFile file = await _controller.takePicture();
      
      // Process the image
      File imageFile = File(file.path);
      List<int> rawImageBytes = await imageFile.readAsBytes();
      Uint8List imageBytes = Uint8List.fromList(rawImageBytes);
      img.Image? capturedImage = img.decodeImage(imageBytes);
      
      if (capturedImage == null) {
        print('Error: Could not decode the captured frame');
        return;
      }
      
      // Resize if needed (to reduce image size for upload)
      if (capturedImage.width > 640 || capturedImage.height > 640) {
        capturedImage = img.copyResize(
          capturedImage, 
          width: capturedImage.width > capturedImage.height ? 640 : null,
          height: capturedImage.height >= capturedImage.width ? 640 : null,
        );
      }
      
      // Encode as JPG with lower quality for frames
      List<int> jpgBytes = img.encodeJpg(capturedImage, quality: 75);
      
      // Convert to base64
      String base64Image = base64Encode(jpgBytes);
      
      // Send via WebSocket
      if (_framesChannel != null && _isFramesConnected) {
        _framesChannel!.sink.add(jsonEncode({
          'type': 'frame',
          'session_id': _sessionId,
          'timestamp': DateTime.now().toIso8601String(),
          'frame_id': const Uuid().v4(), // Generate unique ID for each frame
          'image_data': base64Image,
        }));
      }
    } catch (e) {
      print('Error capturing frame: $e');
    }
  }
  
  // Add a message to the chat
  void _addMessage(String text, bool isUser, {String? imagePath}) {
    setState(() {
      _messages.add(ChatMessage(
        text: text,
        isUser: isUser,
        timestamp: DateTime.now(),
        imageUrl: imagePath,
      ));
    });
    
    // Scroll to bottom after state updates
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }
  
  // Initialize speech recognition
  Future<void> _initSpeech() async {
    bool available = await _speech.initialize(
      onStatus: (status) {
        if (status == 'done') {
          setState(() {
            _isListening = false;
            _startPeriodicFrameCapture();
          });
          // Stop frame capture when speech is done
          if (!_isTyping) {
            _stopPeriodicFrameCapture();
          }
          // Auto-send query when speech recognition is complete
          _sendQueryToServer();
        }
      },
      onError: (errorNotification) {
        setState(() {
          _isListening = false;
        });
        print('Speech to text error: $errorNotification');
        if (!_isTyping) {
          _stopPeriodicFrameCapture();
        }
      },
    );
    if (!available) {
      print('Speech to text not available');
    }
  }
  
  // Send query to server through query/response channel
  void _sendQueryToServer() {
    if (_questionController.text.isEmpty) return;
    
    // Add user message to chat
    _addMessage(_questionController.text, true);
    
    // Send through WebSocket
    if (_queryResponseChannel != null && _isQueryResponseConnected) {
      _queryResponseChannel!.sink.add(jsonEncode({
        'type': 'query',
        'session_id': _sessionId,
        'question': _questionController.text,
        'timestamp': DateTime.now().toIso8601String(),
      }));
      
      setState(() {
        _isProcessing = true;
      });
    } else {
      _addMessage('Error: Unable to send query - not connected to server', false);
    }
    
    // Clear the question field
    _questionController.text = '';
  }
  
  // Initialize text to speech
  Future<void> _initTts() async {
    await _flutterTts.setLanguage('en-US');
    await _flutterTts.setSpeechRate(0.5);
    await _flutterTts.setVolume(1.0);
    await _flutterTts.setPitch(1.0);
    
    _flutterTts.setCompletionHandler(() {
      setState(() {
        _isSpeaking = false;
      });
    });
  }
  
  // Start listening for speech
  void _startListening() async {
    if (!_isListening) {
      bool available = await _speech.initialize();
      if (available) {
        setState(() {
          _isListening = true;
        });
        
        // Start periodic frame capture when listening starts
        _startPeriodicFrameCapture();
        
        await _speech.listen(
          onResult: (result) {
            setState(() {
              _questionController.text = result.recognizedWords;
            });
          },
        );
      }
    }
  }
  
  // Stop listening for speech
  void _stopListening() {
    _speech.stop();
    setState(() {
      _isListening = false;
    });
    
    // When manually stopping listening, check if we should process
    if (_questionController.text.isNotEmpty) {
      _sendQueryToServer();
    }
    
    // Stop frame capture if not typing
    if (!_isTyping) {
      _stopPeriodicFrameCapture();
    }
  }
  
  // Speak text
  Future<void> _speak(String text) async {
    if (text.isNotEmpty) {
      setState(() {
        _isSpeaking = true;
      });
      await _flutterTts.speak(text);
    }
  }
  
  // Stop speaking
  Future<void> _stopSpeaking() async {
    await _flutterTts.stop();
    setState(() {
      _isSpeaking = false;
    });
  }

  @override
  void dispose() {
    // Dispose of controllers and timers
    _controller.dispose();
    _questionController.dispose();
    _scrollController.dispose();
    _queryResponsePingTimer?.cancel();
    _queryResponseReconnectTimer?.cancel();
    _framesPingTimer?.cancel();
    _framesReconnectTimer?.cancel();
    _frameCaptureTimer?.cancel();
    _typingTimer?.cancel();
    _connectivitySubscription?.cancel();
    _disconnectWebSockets();
    _flutterTts.stop();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  // Send a text message
  void _sendMessage() {
    if (_questionController.text.trim().isEmpty) return;
    
    // Start frame capture if not already capturing
    if (!_isCapturingFrames) {
      _startPeriodicFrameCapture();
    }
    
    // Send the query to server
    _sendQueryToServer();
  }
  
  // Format timestamp to time string (HH:MM)
  String _formatTime(DateTime timestamp) {
    final String hour = timestamp.hour.toString().padLeft(2, '0');
    final String minute = timestamp.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[200],
      appBar: AppBar(
        title: Text('Wizard-AI'),
        backgroundColor: const Color.fromARGB(255, 63, 137, 221),
        elevation: 0,
        actions: [
          // Sound control button
          if (_isSpeaking)
            IconButton(
              icon: Icon(Icons.volume_off),
              onPressed: _stopSpeaking,
              tooltip: 'Stop Speaking',
            ),
        ],
      ),
      body: SafeArea(  // Added SafeArea to handle system UI intrusions properly
        child: Stack(
          fit: StackFit.expand,  // Make the stack fill the available space
          children: [
            // Camera area (75% of screen)
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              height: MediaQuery.of(context).size.height * 0.75,
              child: _buildCameraPreview(),
            ),
            
            // Chat area (25% of screen)
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              height: MediaQuery.of(context).size.height * 0.25,
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Messages area
                    // Messages area
                    Expanded(
                      child: _messages.isEmpty
                        ? Center(
                            child: Text(
                              'Start your visual conversation!',
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.grey[700],
                              ),
                            ),
                          )
                        : ListView.builder(
                            controller: _scrollController,
                            padding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                            itemCount: _messages.length,
                            itemBuilder: (context, index) {
                              final message = _messages[index];
                              return _buildMessageBubble(message);
                            },
                          ),
                    ),
                    
                    // Message input area
                    Container(
                      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      color: Colors.white,
                      child: Row(
                        children: [
                          // Mic button
                          Container(
                            decoration: BoxDecoration(
                              color: _isListening 
                                ? Colors.red 
                                : const Color.fromARGB(255, 63, 137, 221),
                              shape: BoxShape.circle,
                            ),
                            child: IconButton(
                              icon: Icon(
                                _isListening ? Icons.mic : Icons.mic_none,
                                color: Colors.white,
                              ),
                              onPressed: _isListening ? _stopListening : _startListening,
                            ),
                          ),
                          SizedBox(width: 8),
                          
                          // Text input field
                          Expanded(
                            child: TextField(
                              controller: _questionController,
                              decoration: InputDecoration(
                                hintText: 'Ask me what you see...',
                                border: OutlineInputBorder(
                                  borderRadius: BorderRadius.circular(20),
                                  borderSide: BorderSide.none,
                                ),
                                filled: true,
                                fillColor: Colors.grey[200],
                                contentPadding: EdgeInsets.symmetric(
                                  horizontal: 16, 
                                  vertical: 8,
                                ),
                              ),
                              minLines: 1,
                              maxLines: 3,
                              textCapitalization: TextCapitalization.sentences,
                              onChanged: (text) {
                                // Text changed, update typing status through listener
                              },
                              onSubmitted: (text) {
                                if (text.trim().isNotEmpty) {
                                  _sendMessage();
                                }
                              },
                            ),
                          ),
                          SizedBox(width: 8),
                          
                          // Send button
                          Container(
                            decoration: BoxDecoration(
                              color: const Color.fromARGB(255, 63, 137, 221),
                              shape: BoxShape.circle,
                            ),
                            child: IconButton(
                              icon: Icon(
                                Icons.send,
                                color: Colors.white,
                              ),
                              onPressed: _sendMessage,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            
            // Connection and processing status overlay
            if (!_isQueryResponseConnected || !_isFramesConnected || _isProcessing)
              Positioned.fill(
                child: Container(
                  color: Colors.black54, // Semi-transparent overlay
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(
                          color: Colors.white,
                        ),
                        SizedBox(height: 16),
                        Text(
                          !_isQueryResponseConnected || !_isFramesConnected
                            ? 'Connecting to server...'
                            : 'Processing your request...',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // Build camera preview widget
  Widget _buildCameraPreview() {
    if (_cameraErrorMessage.isNotEmpty) {
      // Show error message if camera initialization failed
      return Container(
        color: Colors.black,
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.error_outline, size: 60, color: Colors.red),
                SizedBox(height: 16),
                Text(
                  'Camera Error',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  _cameraErrorMessage,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white,
                  ),
                ),
                SizedBox(height: 24),
                ElevatedButton(
                  onPressed: () {
                    // Attempt to reinitialize camera
                    _initializeCamera();
                  },
                  child: Text('Retry'),
                ),
              ],
            ),
          ),
        ),
      );
    }
    
    // Show loading spinner while camera initializes
    if (!_cameraInitialized) {
      return Container(
        color: Colors.black,
        child: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }
    
    // Show camera preview when ready
    return Container(
      color: Colors.black,
      child: Stack(
        children: [
          // Camera preview
          ClipRRect(
            borderRadius: BorderRadius.vertical(bottom: Radius.circular(20)),
            child: Center(
              child: AspectRatio(
                aspectRatio: _controller.value.aspectRatio,
                child: CameraPreview(_controller),
              ),
            ),
          ),
          
          // Status indicators
          Positioned(
            top: 20,
            right: 20,
            child: Column(
              children: [
                // Frame capture indicator
                if (_isCapturingFrames)
                  Container(
                    padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.fiber_manual_record,
                          color: Colors.red,
                          size: 12,
                        ),
                        SizedBox(width: 4),
                        Text(
                          'Live',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                  
                SizedBox(height: 8),
                  
                // Connection status indicator
                Container(
                  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        _isQueryResponseConnected && _isFramesConnected
                          ? Icons.wifi
                          : Icons.wifi_off,
                        color: _isQueryResponseConnected && _isFramesConnected
                          ? Colors.green
                          : Colors.orange,
                        size: 12,
                      ),
                      SizedBox(width: 4),
                      Text(
                        _isQueryResponseConnected && _isFramesConnected
                          ? 'Connected'
                          : 'Connecting',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          
          // Custom capture button
          Positioned(
            bottom: 20,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _captureImage,
                child: Container(
                  width: 70,
                  height: 70,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.3),
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: Container(
                      width: 60,
                      height: 60,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  // Capture an image and send it for analysis
  Future<void> _captureImage() async {
    if (!_cameraInitialized) return;
    
    setState(() {
      _isProcessing = true;
    });
    
    try {
      // Ensure camera is initialized
      await _initializeControllerFuture;
      
      // Capture image
      final XFile file = await _controller.takePicture();
      
      // Read image file
      File imageFile = File(file.path);
      List<int> imageBytes = await imageFile.readAsBytes();
      
      // Process image to proper size
      img.Image? capturedImage = img.decodeImage(Uint8List.fromList(imageBytes));
      if (capturedImage == null) {
        throw Exception('Failed to decode captured image');
      }
      
      // Resize if needed
      if (capturedImage.width > 1200 || capturedImage.height > 1200) {
        capturedImage = img.copyResize(
          capturedImage,
          width: capturedImage.width > capturedImage.height ? 1200 : null,
          height: capturedImage.height >= capturedImage.width ? 1200 : null,
        );
      }
      
      // Encode as JPG with better quality for analysis
      List<int> jpgBytes = img.encodeJpg(capturedImage, quality: 90);
      String base64Image = base64Encode(jpgBytes);
      
      // Add user message with image
      _addMessage(
        _questionController.text.isEmpty 
          ? 'Analyze this image' 
          : _questionController.text, 
        true, // User message
        imagePath: file.path,
      );
      
      // Send HTTP request for analysis
      // We're using HTTP for the captured image analysis instead of WebSocket due to potential size
      var response = await http.post(
        Uri.parse('$_serverUrl/analyze'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'session_id': _sessionId,
          'image': base64Image,
          'question': _questionController.text.isEmpty 
            ? 'Analyze this image and describe what you see' 
            : _questionController.text,
        }),
      );
      
      // Clear question field
      _questionController.text = '';
      
      // Parse response
      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        if (data['status'] == 'success') {
          // Add AI response to messages
          _addMessage(
            data['analysis'], 
            false, // AI message
          );
          
          // Auto-speak the response
          _speak(data['analysis']);
        } else {
          // Add error message
          _addMessage(
            'Error: ${data["error"] ?? "Analysis failed"}',
            false, // AI message
          );
        }
      } else {
        // Handle HTTP error
        _addMessage(
          'Error: Server returned status code ${response.statusCode}',
          false, // AI message
        );
      }
    } catch (e) {
      print('Error capturing image: $e');
      _addMessage(
        'Error: Failed to capture and analyze image',
        false, // AI message
      );
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }
  
  // Build message bubble widget
  Widget _buildMessageBubble(ChatMessage message) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: message.isUser 
          ? MainAxisAlignment.end 
          : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Avatar for AI messages
          if (!message.isUser)
            CircleAvatar(
              radius: 16,
              backgroundColor: const Color.fromARGB(255, 63, 137, 221),
              child: Icon(
                Icons.assistant,
                color: Colors.white,
                size: 18,
              ),
            ),
            
          SizedBox(width: message.isUser ? 0 : 8),
          
          // Message content
          Flexible(
            child: Container(
              padding: EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: message.isUser 
                  ? const Color.fromARGB(255, 63, 137, 221)
                  : Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 2,
                    offset: Offset(0, 1),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Display image if present
                  if (message.imageUrl != null)
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(
                        File(message.imageUrl!),
                        width: 200,
                        height: 150,
                        fit: BoxFit.cover,
                      ),
                    ),
                    
                  if (message.imageUrl != null)
                    SizedBox(height: 8),
                    
                  // Message text
                  Text(
                    message.text,
                    style: TextStyle(
                      color: message.isUser ? Colors.white : Colors.black87,
                    ),
                  ),
                  
                  // Message timestamp
                  Padding(
                    padding: EdgeInsets.only(top: 4),
                    child: Text(
                      _formatTime(message.timestamp),
                      style: TextStyle(
                        color: message.isUser 
                          ? Colors.white.withOpacity(0.7) 
                          : Colors.black54,
                        fontSize: 10,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          SizedBox(width: message.isUser ? 8 : 0),
          
          // Avatar for user messages
          if (message.isUser)
            CircleAvatar(
              radius: 16,
              backgroundColor: Colors.grey[400],
              child: Icon(
                Icons.person,
                color: Colors.white,
                size: 18,
              ),
            ),
        ],
      ),
    );
  }
}