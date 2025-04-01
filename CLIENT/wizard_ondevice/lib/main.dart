import 'package:flutter/material.dart';
import 'package:web_socket_channel/io.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:deepgram_speech_to_text/deepgram_speech_to_text.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:record/record.dart';
import 'package:camera/camera.dart';
import 'dart:typed_data';
import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'package:image/image.dart' as img;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  
  const MyApp({Key? key, required this.cameras}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ChatScreen(cameras: cameras),
    );
  }
}

class ChatScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  
  const ChatScreen({Key? key, required this.cameras}) : super(key: key);
  
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> with WidgetsBindingObserver {
  // WebSocket and TTS
  late WebSocketChannel frameChannel;
  late WebSocketChannel  queryResponseChannel;
  late Deepgram _deepgramTts;
  final TextEditingController _textController = TextEditingController();
  final List<Map<String, String>> _messages = [];

  // Audio Components
  late Deepgram _deepgramStt;
  final _audioRecorder = AudioRecorder(); // Using AudioRecorder from record 6.0.0
  bool _isRecording = false;
  bool _isTranscribing = false;
  String _recordingPath = '';
  DateTime? _lastSpeechTime;
  StreamSubscription? _deepgramStreamSubscription;
  StreamSubscription? _amplitudeSubscription;

  // Camera Components
  late CameraController _cameraController;
  bool _isCameraReady = false;
  Uint8List? _lastFrame;
  Timer? _frameTimer;

  // Speech Detection
  bool _isSpeaking = false;
  Timer? _speechEndTimer;
  final double _speechThreshold = -30.0; // dBFS
  final Duration _speechEndDelay = Duration(seconds: 1);

  // Motion Detection
  final double _motionThreshold = 10.0;
  int _skippedFrames = 0;
  final int _maxSkips = 3;
  img.Image? _lastProcessedFrame;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeSystem();
  }

  Future<void> _initializeSystem() async {
    await _initializeCamera();
    _initializeDeepgram();
    _initializeWebSockets();
  }

  Future<void> _initializeCamera() async {
    try {
      _cameraController = CameraController(
        widget.cameras.firstWhere((c) => c.lensDirection == CameraLensDirection.front),
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await _cameraController.initialize();
      setState(() => _isCameraReady = true);
    } catch (e) {
      print("Camera error: $e");
    }
  }

  void _initializeDeepgram() {
    _deepgramStt = Deepgram('YOUR_DEEPGRAM_API_KEY', baseQueryParams: {
      'model': 'nova-2-general',
      'language': 'en',
      'detect_language': false,
      'punctuate': true,
      'encoding': 'linear16',
      'sample_rate': 16000,
    });

    _deepgramTts = Deepgram('YOUR_DEEPGRAM_API_KEY', baseQueryParams: {
      'model': 'aura-asteria-en',
      'encoding': 'linear16',
      'sample_rate': 16000,
    });
  }

  void _initializeWebSockets() {
    frameChannel = IOWebSocketChannel.connect('ws://d5b2f9g4-5678-90ef-ghij-klmnopqrstuv.ngrok.io/frames');
    queryResponseChannel = IOWebSocketChannel.connect('ws://d5b2f9g4-5678-90ef-ghij-klmnopqrstuv.ngrok.io/query_response');
    
    queryResponseChannel.stream.listen((message) {
      setState(() => _messages.add({"bot": message}));
      _speakText(message);
    });
  }

  Future<void> _speakText(String text) async {
    try {
      final result = await _deepgramTts.speak.text(text);
      // Implement audio playback here using a package like just_audio
    } catch (e) {
      print("TTS error: $e");
    }
  }

  Future<void> _startVoiceInteraction() async {
    if (!_isCameraReady) return;

    final dir = await getTemporaryDirectory();
    _recordingPath = "${dir.path}/recording.wav";

    // First check permissions
    if (await _audioRecorder.hasPermission() == false) {
      print("Audio recording permission not granted");
      return;
    }

    // Configure the recorder
    final config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,  // Use PCM for highest quality
      bitRate: 16000,
      sampleRate: 16000,
      numChannels: 1,
    );

    // Start recording
    try {
      await _audioRecorder.start(config, path: _recordingPath);
      setState(() => _isRecording = true);
    } catch (e) {
      print("Error starting recording: $e");
      return;
    }

    // Listen to amplitude changes for VAD - updated for version 6.0.0
    _amplitudeSubscription = Stream.periodic(Duration(milliseconds: 100))
        .asyncMap((_) async {
          final amplitude = await _audioRecorder.getAmplitude();
          return amplitude.current;
        })
        .listen((amplitude) {
          final dB = 20 * math.log(amplitude / 32767); // Convert to dBFS
          _handleVoiceActivity(dB);
        });

    // For Deepgram streaming, we'll need to use a different approach
    // since we can't directly access the audio stream
    _startDeepgramProcessing();

    _captureAndSendFrame(isFirstFrame: true);
    _frameTimer = Timer.periodic(Duration(milliseconds: 200), (_) {
      if (_isSpeaking) _handleFrameCapture();
    });
  }

  void _startDeepgramProcessing() {
    // Since we can't access the live stream directly, we'll implement
    // a periodic processing approach
    Timer.periodic(Duration(milliseconds: 500), (timer) async {
      if (!_isRecording) {
        timer.cancel();
        return;
      }

      try {
        // This is a workaround - in a real app, you'd want to use
        // a streaming solution directly if possible
        if (await File(_recordingPath).exists()) {
          final bytes = await File(_recordingPath).readAsBytes();
          if (bytes.isNotEmpty) {
            // Process the audio data with Deepgram
            final result = await _deepgramStt.preRecorded.transcribe(
              audio: bytes,
              mimetype: 'audio/wav',
            );
            
            if (result.transcript != null && result.transcript!.isNotEmpty) {
              setState(() {
                _textController.text = result.transcript!;
                _isTranscribing = true;
              });
            }
          }
        }
      } catch (e) {
        print("Error processing audio: $e");
      }
    });
  }

  void _handleVoiceActivity(double dB) {
    final isSpeechDetected = dB > _speechThreshold;
    final now = DateTime.now();

    if (isSpeechDetected) {
      _lastSpeechTime = now;
      if (!_isSpeaking) {
        setState(() => _isSpeaking = true);
        _captureAndSendFrame(isFirstFrame: true);
      }
    } else if (_lastSpeechTime != null && 
              now.difference(_lastSpeechTime!) > _speechEndDelay && 
              _isSpeaking) {
      setState(() => _isSpeaking = false);
      _captureAndSendFrame(isLastFrame: true);
    }
  }

  Future<void> _handleFrameCapture() async {
    if (!_isSpeaking || !_isCameraReady) return;

    final shouldCapture = await _shouldCaptureFrame();
    if (shouldCapture) {
      await _captureAndSendFrame();
    } else {
      _skippedFrames++;
    }
  }

  Future<bool> _shouldCaptureFrame() async {
    if (_skippedFrames >= _maxSkips) return true;

    try {
      final currentFrame = await _getCurrentFrame();
      final currentImage = img.decodeImage(currentFrame);
      
      if (currentImage == null) return false;
      if (_lastProcessedFrame == null) {
        _lastProcessedFrame = currentImage;
        return true;
      }

      final motionScore = _calculateFrameDifference(_lastProcessedFrame!, currentImage);
      _lastProcessedFrame = currentImage;
      return motionScore > _motionThreshold;
    } catch (e) {
      print("Frame processing error: $e");
      return false;
    }
  }

  double _calculateFrameDifference(img.Image previous, img.Image current) {
    if (previous.width != current.width || previous.height != current.height) {
      return 0.0;
    }

    double diff = 0.0;
    for (int y = 0; y < previous.height; y++) {
      for (int x = 0; x < previous.width; x++) {
        final prevPixel = previous.getPixel(x, y);
        final currPixel = current.getPixel(x, y);
        diff += (currPixel.r - prevPixel.r).abs() +
                (currPixel.g - prevPixel.g).abs() +
                (currPixel.b - prevPixel.b).abs();
      }
    }
    return diff / (previous.width * previous.height * 3 * 255);
  }

  Future<Uint8List> _getCurrentFrame() async {
    try {
      final image = await _cameraController.takePicture();
      final file = File(image.path);
      final bytes = await file.readAsBytes();
      await file.delete();
      return bytes;
    } catch (e) {
      print("Error capturing frame: $e");
      return Uint8List(0);
    }
  }

  Future<void> _captureAndSendFrame({
    bool isFirstFrame = false, 
    bool isLastFrame = false
  }) async {
    try {
      final frame = await _getCurrentFrame();
      _lastFrame = frame;
      _skippedFrames = 0;
      
      final metadata = {
        'first_frame': isFirstFrame,
        'last_frame': isLastFrame,
        'timestamp': DateTime.now().millisecondsSinceEpoch,
      };
      
      frameChannel.sink.add(jsonEncode({
        'image': base64Encode(frame),
        'meta': metadata,
      }));
    } catch (e) {
      print("Frame capture error: $e");
    }
  }

  Future<void> _stopVoiceInteraction() async {
    _frameTimer?.cancel();
    _amplitudeSubscription?.cancel();
    
    // Stop recording using the updated API
    if (_isRecording) {
      try {
        await _audioRecorder.stop();
      } catch (e) {
        print("Error stopping recording: $e");
      }
    }
    
    setState(() {
      _isRecording = false;
    });

    // Send the final transcription as a message
    if (_textController.text.isNotEmpty) {
      queryResponseChannel.sink.add(_textController.text);
      setState(() => _messages.add({"user": _textController.text}));
      _textController.clear();
    }

    setState(() => _isTranscribing = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Speech-First Assistant")),
      body: Column(
        children: [
          if (_isCameraReady) 
            SizedBox(
              height: 120,
              child: CameraPreview(_cameraController),
            ),
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (ctx, i) => ChatBubble(
                text: _messages[i].values.first,
                isUser: _messages[i].containsKey("user"),
              ),
            ),
          ),
          if (_isTranscribing)
            LinearProgressIndicator(minHeight: 2),
          Padding(
            padding: EdgeInsets.all(8),
            child: Row(
              children: [
                IconButton(
                  icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                  color: _isRecording ? Colors.red : Colors.blue,
                  onPressed: _isRecording ? _stopVoiceInteraction : _startVoiceInteraction,
                ),
                Expanded(
                  child: TextField(
                    controller: _textController,
                    decoration: InputDecoration(hintText: "Message..."),
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.send),
                  onPressed: () {
                    if (_textController.text.isNotEmpty) {
                      queryResponseChannel.sink.add(_textController.text);
                      setState(() => _messages.add({"user": _textController.text}));
                      _textController.clear();
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _audioRecorder.dispose();
    _cameraController.dispose();
    frameChannel.sink.close();
    queryResponseChannel.sink.close();
    _deepgramStreamSubscription?.cancel();
    _amplitudeSubscription?.cancel();
    super.dispose();
  }
}

extension on Deepgram {
  get preRecorded => null;
}

class ChatBubble extends StatelessWidget {
  final String text;
  final bool isUser;
  
  const ChatBubble({required this.text, this.isUser = false});
  
  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        padding: EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue : Colors.grey[300],
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(text),
      ),
    );
  }
}