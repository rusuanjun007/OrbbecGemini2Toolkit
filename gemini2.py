import queue
import threading
import time

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

        # Visualization thread attributes
        self.vis_queue = queue.Queue(maxsize=2)  # Small queue for viz frames
        self.vis_thread = None
        self.vis_running = False

    def is_connected(self):
        """
        Check if a camera is connected and initialized properly

        Returns:
            bool: True if camera is connected and initialized
        """
        return (
            self.device is not None
            and self.pipeline is not None
            and self.color_stream_profile is not None
        )

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

    def init_camera(self):
        """
        Initialize the camera.
        :return: None
        """
        if not self.is_connected():
            print("Cannot initialize: No camera connected")
            return False

        try:
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
            return True

        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False

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
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()  # Remove oldest
                        except queue.Empty:
                            pass
                    try:
                        self.frame_buffer.put(frame_data, block=False)
                    except queue.Full:
                        pass  # Skip frame if buffer still full
            except Exception as e:
                print(f"Error in acquisition thread: {e}")
                time.sleep(0.1)  # Back off on errors

    def get_latest_frame(self):
        """
        Get the most recent frame from the buffer, dropping old frames if needed.
        """
        if self.frame_buffer.empty():
            return None

        # Skip all but the newest frame if we're falling behind
        with self.lock:
            frames_to_skip = self.frame_buffer.qsize() - 1
            for _ in range(frames_to_skip):
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    break

            try:
                return self.frame_buffer.get_nowait()
            except queue.Empty:
                return None

    def get_status(self):
        """
        Get the current status of the camera system.

        Returns:
            dict: Status information including:
                - frame_buffer_usage: Current buffer utilization (0.0-1.0)
                - acquisition_active: Whether acquisition thread is running
                - pipeline_active: Whether pipeline is running
        """
        buffer_size = self.frame_buffer.qsize()
        buffer_capacity = self.frame_buffer.maxsize

        return {
            "frame_buffer_usage": buffer_size / buffer_capacity
            if buffer_capacity > 0
            else 0,
            "acquisition_active": self.acquisition_thread is not None
            and self.acquisition_thread.is_alive(),
            "pipeline_active": self.pipeline.is_started(),
        }

    def start_visualization_thread(self):
        """
        Start the visualization thread.
        """
        if self.vis_thread is not None and self.vis_thread.is_alive():
            return

        self.vis_running = True
        self.vis_thread = threading.Thread(target=self._visualization_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def stop_visualization_thread(self):
        """
        Stop the visualization thread.
        """
        self.vis_running = False
        if self.vis_thread is not None:
            self.vis_thread.join(timeout=1.0)
            self.vis_thread = None

        # Clear the visualization queue
        while not self.vis_queue.empty():
            try:
                self.vis_queue.get_nowait()
            except queue.Empty:
                break

    def _visualization_loop(self):
        """
        Main visualization loop that runs in a separate thread.
        """
        while self.vis_running:
            try:
                # Block with timeout to allow thread to check if it should exit
                frame_data = self.vis_queue.get(timeout=0.1)

                color_image = frame_data["color_image"]
                depth_data = frame_data["depth_data"]
                kwargs = frame_data["kwargs"]

                # Process masks if present
                if "masks" in kwargs:
                    masks = kwargs["masks"]
                    for mask in masks:
                        mask = mask.cpu().numpy()
                        # CHW to HWC
                        mask = np.transpose(mask, (1, 2, 0))
                        # Repeat the mask HWC=1 to HWC=3
                        mask = np.repeat(mask, 3, axis=2)
                        mask = np.where(mask > 0.0, 1.0, 0.0).astype(np.uint8)
                        color_image[mask == 1] = 255

                # Process points if present
                if "points" in kwargs:
                    points = kwargs["points"]
                    for point in points:
                        cv2.circle(
                            color_image, tuple(point.astype(int)), 5, (0, 255, 0), -1
                        )

                # Create visualization for depth data
                depth_image = cv2.normalize(
                    depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

                # Display images
                cv2.imshow("Color Viewer", color_image)
                cv2.imshow("Depth Viewer", depth_image)
                cv2.waitKey(1)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in visualization thread: {e}")
                time.sleep(0.1)  # Back off on errors

    def visualise_frame(self, color_image, depth_data, **kwargs):
        """
        Queue frames for visualization in a separate thread.

        :param color_image: The color image to visualize.
        :param depth_data: The depth data to visualize.
        :param kwargs: Additional keyword arguments.
        """
        # Make copies to avoid modification of original data
        color_copy = color_image.copy()
        depth_copy = depth_data.copy()

        # If visualization thread is not running, start it
        if self.vis_thread is None or not self.vis_thread.is_alive():
            self.start_visualization_thread()

        # If queue is full, remove oldest frame
        if self.vis_queue.full():
            try:
                self.vis_queue.get_nowait()
            except queue.Empty:
                pass

        # Add the frame to the queue
        try:
            frame_data = {
                "color_image": color_copy,
                "depth_data": depth_copy,
                "kwargs": kwargs,
            }
            self.vis_queue.put(frame_data, block=False)
        except queue.Full:
            pass  # Skip frame if queue is full


if __name__ == "__main__":
    # Test valid FPS

    gemini2 = Gemini2()
    gemini2.init_camera()

    start_time = time.time()

    frames_processed = 0
    while True:
        # It is important to sleep for a short time to avoid CPU overload
        time.sleep(0.01)
        frame_data = gemini2.get_latest_frame()
        if frame_data is None:
            continue

        gemini2.visualise_frame(
            frame_data["color_image"],
            frame_data["depth_data"],
            points=np.array([[250.0, 800.0]], dtype=np.float32),
        )
        frames_processed += 1

        # Calculate FPS every 100 frames
        if frames_processed % 100 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"FPS: {100 / elapsed:.2f}")
            start_time = time.time()
