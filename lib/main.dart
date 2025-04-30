import 'dart:io';
import 'dart:typed_data'; // Required for Uint8List/Float32List
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

// --- IMPORTANT ---
// Ensure you have added the following to your pubspec.yaml:
//
// flutter:
//   uses-material-design: true
//   assets:
//     - assets/models/detect.tflite  # Correct path to your TFLite model
//     # - assets/labels.txt # Optional: if you have a labels file
//
// Also, ensure the 'assets/models' folder exists at the root of your project
// and contains the 'detect.tflite' file.
// --- IMPORTANT ---

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Human Detection', // Added title
      theme: ThemeData( // Added basic theme
         primarySwatch: Colors.blue,
         visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: HumanDetectionScreen(),
    );
  }
}

class HumanDetectionScreen extends StatefulWidget {
  const HumanDetectionScreen({super.key});

  @override
  _HumanDetectionScreenState createState() => _HumanDetectionScreenState();
}

class _HumanDetectionScreenState extends State<HumanDetectionScreen> {
  Interpreter? _interpreter; // Make interpreter nullable
  File? _selectedImage;
  List<Map<String, dynamic>> _results = [];
  bool _loadingModel = true;
  String? _errorLoadingModel;
  bool _isDetecting = false; // Flag to indicate detection in progress

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  @override
  void dispose() {
    _interpreter?.close(); // Dispose of the interpreter
    super.dispose();
  }

  Future<void> loadModel() async {
    setState(() {
      _loadingModel = true;
      _errorLoadingModel = null;
    });
    try {
    
      _interpreter = await Interpreter.fromAsset('assets/models/detect.tflite');
      _interpreter?.allocateTensors(); // Pre-allocate tensors
      print('--- Model loaded successfully ---');
      print('Input tensor details: ${_interpreter?.getInputTensors()}');
      print('Output tensor details: ${_interpreter?.getOutputTensors()}');
    } catch (e) {
      print('--- Error loading model: $e ---');
      setState(() {
        _errorLoadingModel = 'Failed to load model. Ensure "assets/models/detect.tflite" exists and is in pubspec.yaml.\nError: $e';
      });
    } finally {
      setState(() {
        _loadingModel = false;
      });
    }
  }

  Future<void> pickImage() async {
    // Prevent picking a new image while detection is running
    if (_isDetecting) return;

    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile == null) return;

    setState(() {
      _selectedImage = File(pickedFile.path);
      _results.clear(); // Clear previous results
      _isDetecting = true; // Set detecting flag
    });

    // Ensure model is loaded before detecting
    if (_interpreter == null) {
       print("Interpreter not initialized!");
       if (mounted){ 
         ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Model not loaded. Cannot detect.')),
         );
       }
       setState(() { _isDetecting = false; }); // Reset flag
       return;
    }

    // Run detection asynchronously
    await detectObjects(File(pickedFile.path));

    // Reset detecting flag after detection is complete (or fails)
    if (mounted) { // Check if widget is still mounted
      setState(() {
         _isDetecting = false;
      });
    }
  }

  Future<void> detectObjects(File imageFile) async {
    if (_interpreter == null) {
      print("Error: Interpreter is null in detectObjects.");
      return;
    }

    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);

    if (image == null) {
      print("Error: Could not decode image.");
      if (mounted) { // Check if widget is still mounted
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Failed to decode image.')),
        );
      }
      return;
    }

    // --- 1. INPUT PREPROCESSING ---
    // Resize the image to the input size the model expects (e.g., 300x300)
    const int modelInputSize = 300;
    final img.Image inputImage = img.copyResize(image, width: modelInputSize, height: modelInputSize);

    // *** FIX: Use Uint8List because the model expects uint8 input ***
    var input = Uint8List(1 * modelInputSize * modelInputSize * 3).reshape([1, modelInputSize, modelInputSize, 3]);

    // *** FIX: Assign raw pixel values (0-255) without normalization ***
    for (int y = 0; y < modelInputSize; y++) {
      for (int x = 0; x < modelInputSize; x++) {
        var pixel = inputImage.getPixel(x, y);
        input[0][y][x][0] = pixel.r.toInt(); // Red (0-255)
        input[0][y][x][1] = pixel.g.toInt(); // Green (0-255)
        input[0][y][x][2] = pixel.b.toInt(); // Blue (0-255)
      }
    }
    print("--- Input data prepared (Uint8List with shape [1, 300, 300, 3]) ---");


    // --- 2. RUN INFERENCE ---
    // Define outputs based on your model's signature (confirmed from logs)
    // Output 0: Locations (Float32) [1, 10, 4]
    // Output 1: Classes (Float32) [1, 10]
    // Output 2: Scores (Float32) [1, 10]
    // Output 3: Number of Detections (Float32) [1]
    const int numDetectionsMax = 10; // Max detections the model outputs

    // Allocate lists for outputs (using Float32 based on logs)
    var outputLocations = List.generate(1, (_) => List.generate(numDetectionsMax, (_) => List.filled(4, 0.0)), growable: false).cast<List<List<double>>>();
    var outputClasses = List.generate(1, (_) => List.filled(numDetectionsMax, 0.0), growable: false).cast<List<double>>();
    var outputScores = List.generate(1, (_) => List.filled(numDetectionsMax, 0.0), growable: false).cast<List<double>>();
    var numDetections = List.filled(1, 0.0, growable: false).cast<double>(); // Shape [1]

    // Map outputs to their indices
    final outputs = <int, Object>{
      0: outputLocations,
      1: outputClasses,
      2: outputScores,
      3: numDetections,
    };

    print("--- Running model inference ---");
    try {
      _interpreter!.runForMultipleInputs([input], outputs); // Pass the Uint8List input
      print("--- Inference complete ---");
    } catch (e) {
      print("--- Error running inference: $e ---");
      if (mounted) { // Check if widget is still mounted
         ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error during detection: $e')),
        );
      }
      return; // Stop processing if inference fails
    }


    // --- 3. PROCESS OUTPUTS ---
    // Get the actual number of detections from the model's output
    // Clamp the value just in case, ensuring it doesn't exceed numDetectionsMax
    final int numDetectionsOutput = numDetections[0].toInt().clamp(0, numDetectionsMax);

    print('Raw Output numDetections: ${numDetections[0]} (used as: $numDetectionsOutput)');

    List<Map<String, dynamic>> tempResults = [];
    const double confidenceThreshold = 0.5; // Minimum confidence score

    for (int i = 0; i < numDetectionsOutput; i++) {
      // Check confidence score first
      if (outputScores[0][i] > confidenceThreshold) {
        int classIndex = outputClasses[0][i].toInt();

        // **IMPORTANT**: Verify class index 0 is 'person' for your model's label map.
        if (classIndex == 0) { // Assuming Class 0 is 'person'
          // Bounding box format [ymin, xmin, ymax, xmax] (normalized 0.0-1.0)
          if (outputLocations[0][i].length == 4) {
            final List<double> bbox = outputLocations[0][i];
             // Ensure coordinates are valid (between 0 and 1)
             if (bbox.every((coord) => coord >= 0.0 && coord <= 1.0)) {
                 print('Found Person (Index 0): Score=${outputScores[0][i]}, Box=$bbox');
                 tempResults.add({
                   'boundingBox': bbox, // [ymin, xmin, ymax, xmax]
                   'score': outputScores[0][i],
                   'classIndex': classIndex,
                 });
             } else {
                 print('Warning: Invalid bounding box coordinates for detection $i: $bbox');
             }
          } else {
            print('Warning: Bounding box data for detection $i has unexpected length: ${outputLocations[0][i].length}.');
          }
        }
        // else {
        //    print('Detected object: Index=$classIndex, Score=${outputScores[0][i]} (Ignoring non-person)');
        // }
      }
    }

    print('--- Processed ${tempResults.length} Person detections meeting threshold ---');

    // Update the state *after* processing is complete
    if (mounted) { // Check if the widget is still in the tree
       setState(() {
         _results = tempResults;
       });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Human Detection POC')),
      body: Column(
        children: [
          Padding( // Add some padding around the button
            padding: const EdgeInsets.all(12.0),
            child: ElevatedButton(
              // Disable button while loading model or detecting
              onPressed: (_loadingModel || _isDetecting) ? null : pickImage,
              child: _isDetecting
                  ? const Row( // Show progress indicator on button when detecting
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        SizedBox(
                           width: 20, height: 20,
                           child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)
                        ),
                        SizedBox(width: 8),
                        Text('Detecting...'),
                      ],
                    )
                  : const Text('Pick Image from Gallery'),
            ),
          ),
          if (_loadingModel)
            const Expanded( // Make loading indicator fill space
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text("Loading Model...")
                  ],
                ),
              ),
            ),
          if (_errorLoadingModel != null)
             Expanded( // Make error message fill space
               child: Center(
                 child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text(
                     _errorLoadingModel!,
                     style: const TextStyle(color: Colors.red, fontWeight: FontWeight.bold),
                     textAlign: TextAlign.center,
                  ),
                 ),
               ),
             ),
          // Show image and results area only if model loaded successfully and no error
          if (!_loadingModel && _errorLoadingModel == null)
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: (_selectedImage == null)
                  ? const Center(child: Text('Pick an image to start detection.'))
                  : LayoutBuilder( // Use LayoutBuilder to get actual render size
                      builder: (context, constraints) {
                        final double containerWidth = constraints.maxWidth;
                        final double containerHeight = constraints.maxHeight;

                        return Stack(
                          alignment: Alignment.center, // Center the image within the stack
                          children: [
                            // Display the selected image
                            Image.file(
                              _selectedImage!,
                              // Fit the image within the container while preserving aspect ratio
                              fit: BoxFit.contain,
                              width: containerWidth,
                              height: containerHeight,
                            ),
                            // Draw bounding boxes on top
                            // Need to get the actual rendered image size and position
                            // This part is complex if using BoxFit.contain and aspect ratios differ.
                            // For simplicity, let's assume the Stack covers the same area as the image
                            // (This might need refinement depending on exact layout needs)
                            ..._results.map((result) {
                              final List<double> bbox = result['boundingBox']; // [ymin, xmin, ymax, xmax]
                              final double score = result['score'];

                              // Scale normalized coordinates to the container size
                              // NOTE: This assumes the image fills the container width or height
                              // after BoxFit.contain scaling. For precise alignment on images with
                              // different aspect ratios than the container, more complex calculations
                              // involving the actual rendered image dimensions are needed.
                              final double yMin = bbox[0] * containerHeight;
                              final double xMin = bbox[1] * containerWidth;
                              final double yMax = bbox[2] * containerHeight;
                              final double xMax = bbox[3] * containerWidth;
                              final double BboxWidth = (xMax - xMin).abs(); // Use abs for safety
                              final double BboxHeight = (yMax - yMin).abs(); // Use abs for safety


                              // Check for valid box dimensions before attempting to draw
                              if (BboxWidth > 0 && BboxHeight > 0 &&
                                  xMin >= 0 && yMin >= 0 && // Ensure top-left is within bounds
                                  (xMin + BboxWidth) <= containerWidth &&
                                  (yMin + BboxHeight) <= containerHeight)
                              {
                                return Positioned(
                                  left: xMin,
                                  top: yMin,
                                  width: BboxWidth,
                                  height: BboxHeight,
                                  child: Container(
                                    decoration: BoxDecoration(
                                      border: Border.all(color: Colors.red, width: 2), // Slightly thinner border
                                    ),
                                    child: Align(
                                      alignment: Alignment.topLeft,
                                      child: Container(
                                         // *** FIX: Use .withAlpha() instead of withOpacity ***
                                         color: Colors.red.withAlpha((0.6 * 255).toInt()), // Calculate alpha from opacity (0-255)
                                         padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                                         child: Text(
                                          'P: ${(score * 100).toStringAsFixed(0)}%', // Shorter label
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 11, // Slightly smaller font
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ),
                                    ),
                                  ),
                                );
                              } else {
                                print('Skipping drawing box outside bounds or with invalid dimensions: xMin=$xMin, yMin=$yMin, W=$BboxWidth, H=$BboxHeight');
                                return Container(); // Invalid box, draw nothing
                              }
                            }).toList(),
                          ],
                        );
                      },
                    ),
              ),
            )
        ],
      ),
    );
  }
}
