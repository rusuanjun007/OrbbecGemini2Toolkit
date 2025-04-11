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
