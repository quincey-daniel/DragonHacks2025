import asyncio
import cv2
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define Pydantic models for object classification
class DetectedObject(BaseModel):
    """Represents a single detected object in the image."""
    class_name: str = Field(description="Name of the detected object class (e.g., 'person', 'car', 'chair')")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)
    bounding_box: Optional[Tuple[int, int, int, int]] = Field(
        None, 
        description="Optional bounding box coordinates as (x, y, width, height)"
    )
    
class ObjectClassification(BaseModel):
    """Full classification results for an image frame."""
    objects: List[DetectedObject] = Field(description="List of all objects detected in the frame")
    timestamp: datetime = Field(default_factory=datetime.now, description="Time when the frame was analyzed")
    summary: str = Field(description="Brief summary of what's seen in the image")
    dominant_objects: List[str] = Field(
        description="List of the most notable or numerous objects in the frame"
    )
    total_objects: int = Field(description="Total number of objects detected")
    scene_type: str = Field(description="General classification of the scene (indoor, outdoor, etc.)")

# Initialize the Gemini model
model = GeminiModel(
    'gemini-2.0-flash', provider=GoogleGLAProvider(api_key=GEMINI_API_KEY)
)

# Configure the agent with our object classification output type
agent = Agent(
    model=model,
    output_type=ObjectClassification,
    system_prompt=(
        "You are an object detection system. Analyze the image and identify all "
        "visible objects. Provide confidence scores based on how certain you are "
        "about each detection. Assign confidence values between 0.0 and 1.0. "
        "For dominant_objects, list only the 2-3 most prominent objects in the scene. "
        "Provide a brief one-sentence summary of what's in the image."
    )
)

# Object detection history
detection_history: List[ObjectClassification] = []

async def process_video_stream():
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            # Convert frame to PNG format in memory
            _, buffer = cv2.imencode('.png', frame)
            frame_bytes = buffer.tobytes()
            
            # Create binary content
            frame_content = BinaryContent(data=frame_bytes, media_type='image/png')
            
            # Pass to agent
            result = await agent.run([
                "Identify and classify all objects in this image", 
                frame_content
            ])
            
            # Store the classification result in our history
            classification = result.output
            detection_history.append(classification)
            
            # Display the frame with detection results
            display_frame = frame.copy()
            
            # Show summary and scene type at the top
            cv2.putText(display_frame, f"Scene: {classification.scene_type}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Summary: {classification.summary[:50]}...", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # List dominant objects
            cv2.putText(display_frame, f"Main objects: {', '.join(classification.dominant_objects)}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display total object count
            cv2.putText(display_frame, f"Objects detected: {classification.total_objects}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # List all detected objects with confidence
            for i, obj in enumerate(classification.objects[:5]):  # Show top 5 objects
                y_pos = 150 + (i * 30)
                cv2.putText(display_frame, 
                            f"{obj.class_name}: {obj.confidence:.2f}", 
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the enhanced frame
            cv2.imshow('Object Classification', display_frame)
            
            # Print details to console
            print(f"\n--- Detection at {classification.timestamp} ---")
            print(f"Scene type: {classification.scene_type}")
            print(f"Summary: {classification.summary}")
            print(f"Dominant objects: {classification.dominant_objects}")
            print(f"Total objects: {classification.total_objects}")
            for obj in classification.objects:
                print(f"  - {obj.class_name}: {obj.confidence:.2f}")
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Save the last classification to a file (optional)
            with open("last_classification.txt", "w") as f:
                f.write(f"Scene: {classification.scene_type}\n")
                f.write(f"Summary: {classification.summary}\n")
                f.write(f"Dominant objects: {', '.join(classification.dominant_objects)}\n")
                f.write(f"Total objects: {classification.total_objects}\n")
                f.write("Detected objects:\n")
                for obj in classification.objects:
                    f.write(f"  - {obj.class_name}: {obj.confidence:.2f}\n")
            
            # Sleep to avoid overwhelming the model API
            await asyncio.sleep(2)
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Print a summary of the session
        print(f"\nSession Summary:")
        print(f"Total frames analyzed: {len(detection_history)}")
        if detection_history:
            all_objects = set()
            for detection in detection_history:
                all_objects.update(obj.class_name for obj in detection.objects)
            print(f"Unique objects detected: {', '.join(sorted(all_objects))}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(process_video_stream())