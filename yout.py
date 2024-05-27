import os
import cv2
from pytube import YouTube

# Set the output directory for downloaded videos and extracted frames
# output_dir = 'path/to/output/directory'
output_dir = 'youtube-videos'
frames_dir = os.path.join(output_dir, 'frames')

# Create the directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

# URLs of the YouTube videos to download
video_urls = [
    # 'https://youtu.be/aTsv6_ZJOnY?si=Li-7LN4VDFKsrh0-'
    'https://youtu.be/CPwaMlPJNJ4?si=RyByVI3GK142zdIh'
    # Add more video URLs as needed
]

# Download videos using pytube
for url in video_urls:
    video = YouTube(url)
    stream = video.streams.get_highest_resolution()
    video_path = stream.download(output_dir)

    # Extract frames from the downloaded video
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video using OpenCV
    video = cv2.VideoCapture(video_path)

    # Initialize frame count
    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        if not ret:
            break

        # Save the frame as an image
        frame_name = f"{video_name}_frame_{frame_count}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video capture object
    video.release()
