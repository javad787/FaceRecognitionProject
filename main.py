import cv2
from face_recognition import FaceRecognition


class VideoCapture:
    """
    This class is responsible for capturing video from a camera and processing each frame.
    """

    def __init__(self):
        """
        Initialize VideoCapture object with a camera and a FaceRecognition object.
        """
        self.video_capture = cv2.VideoCapture(0)
        self.face_recognition = FaceRecognition()

    def run(self):
        """
        Run the video capture loop.
        Capture frames from the camera, process each frame using the FaceRecognition object,
        and display the processed frames in a window.
        Stop capturing when 'q' is pressed.
        """
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                continue

            processed_frame = self.face_recognition.process_frame(frame)

            cv2.imshow("Video", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Entry point of the program.
    Create a VideoCapture object and run the video capture loop.
    """
    video_capture = VideoCapture()
    video_capture.run()