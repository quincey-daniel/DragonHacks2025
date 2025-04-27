import asyncio
import cv2
import time
from pydantic_ai import Agent, BinaryContent
import os
from dotenv import load_dotenv
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = GeminiModel(
    'gemini-2.0-flash', provider=GoogleGLAProvider(api_key=GEMINI_API_KEY)
)

agent = Agent(model)

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
                "What do you see in this camera frame?", 
                frame_content
            ])
            
            # Display the frame with the agent's response
            cv2.putText(frame, f"Agent: {result.output[:50]}...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Camera Feed with Agent Analysis', frame)
            cv2.waitKey(1)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Sleep to avoid overwhelming the model API
            await asyncio.sleep(2)  # Adjust this based on your needs and rate limits
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Run the async function
if __name__ == "__main__":
    asyncio.run(process_video_stream())