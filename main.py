# pip install opencv-python numpy matplotlib pillow

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class FaceDetector:
    """Face detector using OpenCV's Haar Cascade"""
    
    def __init__(self):
        # Load pre-trained Haar Cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load eye cascade for additional detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Load smile cascade
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
    
    def detect_faces(self, image, detect_eyes=False, detect_smile=False):
        """Detect faces in an image"""
        # Convert to grayscale (Haar Cascade works on grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # How much image size is reduced at each scale
            minNeighbors=5,       # How many neighbors each candidate should have
            minSize=(30, 30)      # Minimum face size
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            face_info = {
                'bbox': (x, y, w, h),
                'eyes': [],
                'smile': None
            }
            
            # Optionally detect eyes within face region
            if detect_eyes:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                face_info['eyes'] = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
            
            # Optionally detect smile within face region
            if detect_smile:
                roi_gray = gray[y+h//2:y+h, x:x+w]  # Lower half of face
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                if len(smiles) > 0:
                    face_info['smile'] = True
            
            results.append(face_info)
        
        return results
    
    def draw_detections(self, image, detections, draw_eyes=False):
        """Draw bounding boxes on detected faces"""
        output_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Draw face rectangle
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
            # Add label
            label = "Face"
            if detection.get('smile'):
                label += " (Smiling)"
            
            cv2.putText(output_image, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Draw eyes if detected
            if draw_eyes:
                for (ex, ey, ew, eh) in detection['eyes']:
                    cv2.rectangle(output_image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return output_image


def detect_from_image(image_path, detect_eyes=False, detect_smile=False):
    """Detect faces from an image file"""
    print("=" * 70)
    print(f"DETECTING FACES IN IMAGE: {image_path}")
    print("=" * 70)
    
    # Load image
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            print("Please provide a valid image file.")
            return
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Initialize detector
    detector = FaceDetector()
    
    # Detect faces
    print("\nDetecting faces...")
    detections = detector.detect_faces(image, detect_eyes, detect_smile)
    
    print(f"Found {len(detections)} face(s)!")
    
    # Draw detections
    output_image = detector.draw_detections(image, detections, draw_eyes=detect_eyes)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Detected faces
    axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Detected: {len(detections)} Face(s)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('face_detection_result.png', dpi=150, bbox_inches='tight')
    print("\nResult saved to 'face_detection_result.png'")
    plt.show()
    
    # Print detection details
    print("\nDetection Details:")
    print("-" * 70)
    for i, detection in enumerate(detections, 1):
        x, y, w, h = detection['bbox']
        print(f"Face {i}:")
        print(f"  Position: (x={x}, y={y})")
        print(f"  Size: {w}x{h} pixels")
        if detect_eyes:
            print(f"  Eyes detected: {len(detection['eyes'])}")
        if detect_smile and detection.get('smile'):
            print(f"  Smile: Yes")


def detect_from_webcam(duration=10):
    """Detect faces from webcam feed"""
    print("=" * 70)
    print("WEBCAM FACE DETECTION")
    print("=" * 70)
    print("\nStarting webcam...")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("-" * 70)
    
    # Initialize detector
    detector = FaceDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("\nWebcam not available. Use detect_from_image() instead.")
        return
    
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect faces
        detections = detector.detect_faces(frame, detect_eyes=True)
        
        # Draw detections
        output_frame = detector.draw_detections(frame, detections, draw_eyes=True)
        
        # Add info text
        cv2.putText(output_frame, f"Faces: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_frame, "Press 'q' to quit, 's' to save", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Face Detection - Webcam', output_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f'webcam_capture_{frame_count}.jpg'
            cv2.imwrite(filename, output_frame)
            print(f"Screenshot saved: {filename}")
            frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


def create_test_image():
    """Create a simple test image with a face-like pattern"""
    print("\nCreating test image...")
    
    # Create a blank image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw a simple face
    # Face circle
    cv2.circle(img, (200, 200), 100, (200, 150, 100), -1)
    
    # Eyes
    cv2.circle(img, (170, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (230, 180), 15, (50, 50, 50), -1)
    
    # Smile
    cv2.ellipse(img, (200, 230), (40, 20), 0, 0, 180, (50, 50, 50), 3)
    
    # Save
    cv2.imwrite('test_face.jpg', img)
    print("Test image created: test_face.jpg")
    
    return 'test_face.jpg'


def batch_detect_faces(image_folder):
    """Detect faces in multiple images"""
    import os
    import glob
    
    print("=" * 70)
    print(f"BATCH FACE DETECTION IN FOLDER: {image_folder}")
    print("=" * 70)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"\nFound {len(image_files)} image(s)")
    print("-" * 70)
    
    # Initialize detector
    detector = FaceDetector()
    
    total_faces = 0
    
    for img_path in image_files:
        try:
            # Load and detect
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            detections = detector.detect_faces(image)
            total_faces += len(detections)
            
            print(f"{os.path.basename(img_path)}: {len(detections)} face(s)")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("-" * 70)
    print(f"Total faces detected: {total_faces}")


def main():
    """Main function with different modes"""
    
    choice = input("\nEnter choice (1-4) or press Enter to use test image: ").strip()
    
    if choice == "1":
        image_path = input("Enter image path: ").strip()
        detect_from_image(image_path, detect_eyes=True, detect_smile=True)
    
    elif choice == "2":
        print("\nStarting webcam detection...")
        detect_from_webcam()
    
    elif choice == "3":
        test_image = create_test_image()
        detect_from_image(test_image, detect_eyes=True)
    
    elif choice == "4":
        folder = input("Enter folder path: ").strip()
        batch_detect_faces(folder)
    
    else:
        # Default: sample image
        print("\nUsing default mode: Creating test image...")
        test_image = create_test_image()
        detect_from_image(test_image, detect_eyes=True)
    
    print("\n" + "=" * 70)
    print("✓ Face detection complete!")
    print("=" * 70)
    print("\nQuick usage examples:")
    print("  detector = FaceDetector()")
    print("  image = cv2.imread('photo.jpg')")
    print("  faces = detector.detect_faces(image)")
    print("  result = detector.draw_detections(image, faces)")


if __name__ == "__main__":
    main()