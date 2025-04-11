from typing import Any, Optional, Union

import cv2
import numpy as np
from pyorbbecsdk import FormatConvertFilter, OBConvertFormat, OBFormat, VideoFrame


def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    yuyv = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return bgr_image


def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    uyvy = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image


def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height : height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4 :].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image


def determine_convert_format(frame: VideoFrame):
    if frame.get_format() == OBFormat.I420:
        return OBConvertFormat.I420_TO_RGB888
    elif frame.get_format() == OBFormat.MJPG:
        return OBConvertFormat.MJPG_TO_RGB888
    elif frame.get_format() == OBFormat.YUYV:
        return OBConvertFormat.YUYV_TO_RGB888
    elif frame.get_format() == OBFormat.NV21:
        return OBConvertFormat.NV21_TO_RGB888
    elif frame.get_format() == OBFormat.NV12:
        return OBConvertFormat.NV12_TO_RGB888
    elif frame.get_format() == OBFormat.UYVY:
        return OBConvertFormat.UYVY_TO_RGB888
    else:
        return None


def frame_to_rgb_frame(frame: VideoFrame) -> Union[Optional[VideoFrame], Any]:
    if frame.get_format() == OBFormat.RGB:
        return frame
    convert_format = determine_convert_format(frame)
    if convert_format is None:
        print("Unsupported format")
        return None
    print("covert format: {}".format(convert_format))
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        print("Convert {} to RGB failed".format(frame.get_format()))
    return rgb_frame


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


def calculate_FoV(Z):
    def _h_fov(Z):
        """
        Calculate the field of view (FOV) based on the given parameters.

        Parameters:
        Z (float): The distance from the camera to the object.

        Returns:
        float: The calculated field of view.
        """
        cx = 636.404663
        fx = 613.787292
        B = 0.05
        width = 1280

        # Calculate the field of view using the provided formula
        # Note: The formula is derived from the pinhole camera model
        # and assumes a rectangular image sensor.

        # Ensure Z is not zero to avoid division by zero
        if Z == 0:
            raise ValueError("Z must be non-zero to calculate FOV.")

        # Calculate FOV
        # The formula is derived from the pinhole camera model
        # and assumes a rectangular image sensor.
        fov = np.arctan(cx / fx - B / Z) + np.arctan((width - 1 - cx) / fx)
        active_fov = np.arctan(cx / fx) + np.arctan((width - 1 - cx) / fx)

        # Convert radians to degrees
        return fov * 180 / np.pi, active_fov * 180 / np.pi

    def _v_fov(Z):
        """
        Calculate the vertical field of view (FOV) based on the given parameters.

        Parameters:
        Z (float): The distance from the camera to the object.

        Returns:
        float: The calculated vertical field of view.
        """
        cy = 396.343628
        fy = 613.787292
        B = 0.05
        height = 800

        # Ensure Z is not zero to avoid division by zero
        if Z == 0:
            raise ValueError("Z must be non-zero to calculate FOV.")

        # Calculate FOV
        fov = np.arctan(cy / fy - B / Z) + np.arctan((height - 1 - cy) / fy)
        active_fov = np.arctan(cy / fy) + np.arctan((height - 1 - cy) / fy)

        # Convert radians to degrees
        return fov * 180 / np.pi, active_fov * 180 / np.pi

    hfov = _h_fov(Z)
    vfov = _v_fov(Z)
    print(f"Horizontal FoV at {Z}m: {hfov[0]}, Max: {hfov[1]}")
    print(f"Vertical FoV at {Z}m: {vfov[0]}, Max: {vfov[1]}")


def check_intrinsics():
    """
    Check the intrinsic and extrinsic parameters of the camera.

    This function retrieves the intrinsic and extrinsic parameters of the camera
    and prints them to the console.
    """
    from pyorbbecsdk import OBSensorType, Pipeline

    pipeline = Pipeline()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

    # Get color_profile
    color_profile = profile_list.get_default_video_stream_profile()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    # Get depth_profile
    depth_profile = profile_list.get_default_video_stream_profile()

    # Get external parameters
    extrinsic = depth_profile.get_extrinsic_to(color_profile)
    print("extrinsic  {}".format(extrinsic))

    # Get depth inernal parameters
    depth_intrinsics = depth_profile.get_intrinsic()
    print("depth_intrinsics  {}".format(depth_intrinsics))

    # Get depth distortion parameter
    depth_distortion = depth_profile.get_distortion()
    print("depth_distortion  {}".format(depth_distortion))

    # Get color internala parameters
    color_intrinsics = color_profile.get_intrinsic()
    print("color_intrinsics  {}".format(color_intrinsics))

    # Get color distortion parameter
    color_distortion = color_profile.get_distortion()
    print("color_distortion  {}".format(color_distortion))
