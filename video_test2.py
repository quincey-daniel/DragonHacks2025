import asyncio
import cv2
import time
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

from pydantic_ai import Agent, BinaryContent
import os
from dotenv import load_dotenv
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider


@dataclass
class RPiCameraV2Specs:
    """Hardware specifications for the Raspberry Pi Camera Module V2."""
    sensor_name: str = "Sony IMX219"
    still_resolution: Tuple[int, int] = (3280, 2464)  # 8MP
    video_modes: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "full_hd": (1920, 1080, 47),  # 1080p47
        "medium": (1640, 1232, 41),    # 1640 × 1232p41
        "low": (640, 480, 206)         # 640 × 480p206
    })
    focal_length_mm: float = 3.04  # mm
    f_stop: float = 2.0
    horizontal_fov: float = 62.2  # degrees
    vertical_fov: float = 48.8    # degrees

@dataclass
class RPiCameraV2:
    """Handles the Raspberry Pi Camera Module V2."""
    camera_id: int = 0
    resolution: Tuple[int, int] = field(default_factory=lambda: (1920, 1080))
    framerate: int = 30
    specs: RPiCameraV2Specs = field(default_factory=RPiCameraV2Specs)
    
    # Runtime fields
    cap: Optional[cv2.VideoCapture] = None
    
    def __post_init__(self):
        """Initialize camera with correct backend and parameters."""
        try:
            # Try using V4L2 backend specific to Raspberry Pi
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        except Exception:
            # Fall back to default
            self.cap = cv2.VideoCapture(self.camera_id)
            
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera with ID {self.camera_id}")
        
        # Configure the camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        return self.cap.read()
    
    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None

@dataclass
class RPiVideoFrame:
    """Represents a single frame with metadata."""
    frame: np.ndarray
    timestamp: float = field(default_factory=time.time)
    
    def to_binary_content(self, format: str = "png") -> BinaryContent:
        """Convert frame to BinaryContent for PydanticAI."""
        if self.frame is None:
            raise ValueError("Cannot convert empty frame to BinaryContent")
        
        if format.lower() == "png":
            _, buffer = cv2.imencode('.png', self.frame)
            media_type = "image/png"
        elif format.lower() in ["jpeg", "jpg"]:
            _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            media_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        frame_bytes = buffer.tobytes()
        return BinaryContent(data=frame_bytes, media_type=media_type)

# Create a PydanticAI agent
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = GeminiModel(
    'gemini-2.0-flash', provider=GoogleGLAProvider(api_key=GEMINI_API_KEY)
)
agent = Agent(model)

async def process_video_stream():
    # Initialize RPi Camera V2
    camera = RPiCameraV2(
        camera_id=0,
        resolution=(1280, 720),  # HD resolution for better performance
        framerate=30
    )
    
    window_name = 'Camera Feed with Agent Analysis'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Capture frame
            ret, frame = camera.read_frame()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            # Create RPiVideoFrame object
            video_frame = RPiVideoFrame(frame=frame)
            
            # Create binary content using our dataclass method
            frame_content = video_frame.to_binary_content(format="png")
            
            # Pass to agent
            result = await agent.run([
                "What do you see in this camera frame?", 
                frame_content
            ])
            
            # Display the frame with the agent's response
            display_frame = frame.copy()
            
            # Split text into multiple lines if it's long
            text = result.output
            max_chars_per_line = 60
            text_lines = [text[i:i+max_chars_per_line] 
                         for i in range(0, min(len(text), 150), max_chars_per_line)]
            
            for i, line in enumerate(text_lines):
                y_position = 30 + (i * 30)  # 30 pixels per line
                cv2.putText(display_frame, line, (10, y_position), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Sleep to avoid overwhelming the model API
            await asyncio.sleep(2)  # Adjust based on your needs and rate limits
    
    finally:
        # Release resources
        camera.release()
        cv2.destroyAllWindows()

# Run the async function
if __name__ == "__main__":
    asyncio.run(process_video_stream())