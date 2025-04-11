import queue
import threading

import cv2
import numpy as np
from pyorbbecsdk import Config, OBAlignMode, OBSensorType, Pipeline

from utils import frame_to_bgr_image


class Gemini2:
    def __init__(self, buffer_size=5):
        self.pipeline = Pipeline()
        self.config = Config()
        self.device = self.pipeline.get_device()

        # Color setting
        self.color_stream_profile = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )

        # Depth setting
        self.MIN_DEPTH = 20  # 20mm
        self.MAX_DEPTH = 10000  # 10000mm
        self.depth_work_mode_list = self.device.get_depth_work_mode_list()
        self.depth_stream_profile = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        self.depth_sensor = self.device.get_sensor(OBSensorType.DEPTH_SENSOR)
        self.filter_list = self.depth_sensor.get_recommended_filters()

        # Thread-related attributes
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        self.acquisition_thread = None
        self.lock = threading.Lock()

    def start_pipeline(self):
        """
        Start the pipeline.
        """
        self.pipeline.start(self.config)

    def stop_pipeline(self):
        """
        Stop the pipeline.
        """
        self.pipeline.stop()

    def get_depth_work_mode(self):
        """
        Get the current depth work mode.
        :return: The current depth work mode.
        """
        current_depth_work_mode = self.device.get_depth_work_mode()
        print("Current depth work mode: ", current_depth_work_mode)

        for i in range(self.depth_work_mode_list.get_count()):
            depth_work_mode = self.depth_work_mode_list.get_depth_work_mode_by_index(i)
            print(f"{i}. {depth_work_mode}")

    def set_depth_work_mode(self, index):
        """
        Set the current depth work mode.

        :param index: The index of the depth work mode to set.
            0. Unbinned Dense Default
            1. Binned Sparse Default
            2. Unbinned Sparse Default
            3. In-scene Calibration
            4. Obstacle Avoidance
        """
        select_depth_work_mode = self.depth_work_mode_list.get_depth_work_mode_by_index(
            index
        )
        self.device.set_depth_work_mode(select_depth_work_mode.name)
        assert select_depth_work_mode == self.device.get_depth_work_mode()
        print(f"Set depth work mode to {select_depth_work_mode} success!")

    def get_depth_stream_profile(self):
        """
        Get the depth stream profile.
        :return: The depth stream profile.
        """
        profile_count = self.depth_stream_profile.get_count()
        for i in range(profile_count):
            depth_profile = self.depth_stream_profile.get_stream_profile_by_index(i)
            print(f"Depth profile {i}: {depth_profile} {depth_profile.get_format()}")

    def set_depth_stream_profile(self, index):
        """
        Set the depth stream profile.

        :param index: The index of the depth stream profile to set.
        """
        depth_profile = self.depth_stream_profile.get_stream_profile_by_index(index)
        self.config.enable_stream(depth_profile)
        print(
            f"Set depth stream profile to {depth_profile} {depth_profile.get_format()} success!"
        )

    def get_color_steam_profile(self):
        """
        Get the color stream profile.
        :return: The color stream profile.
        """
        profile_count = self.color_stream_profile.get_count()
        for i in range(profile_count):
            color_profile = self.color_stream_profile.get_stream_profile_by_index(i)
            print(f"Color profile {i}: {color_profile} {color_profile.get_format()}")

    def set_color_stream_profile(self, index):
        """
        Set the color stream profile.

        :param index: The index of the color stream profile to set.
        """
        color_profile = self.color_stream_profile.get_stream_profile_by_index(index)
        self.config.enable_stream(color_profile)
        print(
            f"Set color stream profile to {color_profile} {color_profile.get_format()} success!"
        )

    def get_depth_post_filter(self):
        # print filter list
        for i in range(len(self.filter_list)):
            post_filter = self.filter_list[i]
            if post_filter:
                print(
                    f"filter name: {post_filter.get_name()} | enabled {post_filter.is_enabled()}",
                )

    def post_filter(self, depth_frame):
        depth_data_size = depth_frame.get_data()
        if len(depth_data_size) < (
            depth_frame.get_width() * depth_frame.get_height() * 2
        ):
            # depth data is not complete
            return depth_frame

        for i in range(len(self.filter_list)):
            post_filter = self.filter_list[i]
            if post_filter and post_filter.is_enabled() and depth_frame:
                new_depth_frame = post_filter.process(depth_frame)
                depth_frame = new_depth_frame.as_depth_frame()

        return depth_frame

    def start_acquisition_thread(self):
        """
        Start the frame acquisition thread.
        """
        if self.acquisition_thread is not None and self.acquisition_thread.is_alive():
            return

        self.is_running = True
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop)
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()

    def stop_acquisition_thread(self):
        """
        Stop the frame acquisition thread.
        """
        self.is_running = False
        if self.acquisition_thread is not None:
            self.acquisition_thread.join(timeout=1.0)
            self.acquisition_thread = None

        # Clear the frame buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break

    def _acquisition_loop(self):
        """
        Main acquisition loop that runs in a separate thread.
        """
        while self.is_running:
            try:
                frame_data = self.get_one_frame_data()
                if frame_data is not None:
                    # If buffer is full, remove oldest frame
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    # Add new frame to buffer
                    self.frame_buffer.put(frame_data, block=False)
            except Exception as e:
                print(f"Error in acquisition thread: {e}")
                time.sleep(0.01)  # Prevent tight loop if errors occur

    def get_latest_frame(self):
        """
        Get the most recent frame from the buffer.
        """
        if self.frame_buffer.empty():
            return None

        # Get the most recent frame
        with self.lock:
            # Empty the queue except for the last item
            latest_frame = None
            while not self.frame_buffer.empty():
                try:
                    latest_frame = self.frame_buffer.get_nowait()
                except queue.Empty:
                    break

            return latest_frame

    def init_camera(self):
        """
        Initialize the camera.
        :return: None
        """
        # Set the depth work mode
        # Unbinned Dense Default
        self.set_depth_work_mode(0)

        # Set the depth stream
        # <VideoStreamProfile: 1280x800@30> OBFormat.Y16
        self.set_depth_stream_profile(2)

        # Set the color stream
        # <VideoStreamProfile: 1920x1080@30> OBFormat.RGB success!
        self.set_color_stream_profile(1)

        # Set the alignment mode to hardware alignment
        self.config.set_align_mode(OBAlignMode.HW_MODE)
        print("Set align mode to HW_MODE success!")

        # Get post depth filter setting
        self.get_depth_post_filter()

        self.pipeline.start(self.config)

        # Start the frame acquisition thread
        self.start_acquisition_thread()

    def get_one_frame_data(self):
        """
        Get the depth stream.
        :return: The depth stream.
        """
        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                return None

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if depth_frame is None or color_frame is None:
                return None

            # Process color frame
            color_image = frame_to_bgr_image(color_frame)

            # Process depth frame
            self.post_filter(depth_frame)
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where(
                (depth_data > self.MIN_DEPTH) & (depth_data < self.MAX_DEPTH),
                depth_data,
                0,
            )
            return {"color_image": color_image, "depth_data": depth_data}

        except Exception as e:
            print(f"Error: {e}")

    def visualise_frame(self, color_image, depth_data):
        depth_image = cv2.normalize(
            depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        cv2.imshow("Color Viewer", color_image)
        cv2.imshow("Depth Viewer", depth_image)
        cv2.waitKey(1)

    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        self.stop_acquisition_thread()
        self.stop_pipeline()


if __name__ == "__main__":
    # Test valid FPS
    import time

    gemini2 = Gemini2()
    gemini2.init_camera()

    start_time = time.time()

    frames_processed = 0
    while True:
        frame_data = gemini2.get_latest_frame()
        if frame_data is not None:
            gemini2.visualise_frame(frame_data["color_image"], frame_data["depth_data"])
            frames_processed += 1

            # Calculate FPS every 100 frames
            if frames_processed % 100 == 0:
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"FPS: {100 / elapsed:.2f}")
                start_time = time.time()

        if cv2.waitKey(1) == 27:  # ESC key
            break
