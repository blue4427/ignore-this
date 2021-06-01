import pyrealsense2 as rs
import numpy as np
import cv2

DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07","0B3A"]

def find_device_that_supports_advanced_mode() :
    ctx = rs.context()
    devices = ctx.query_devices();
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                return dev

advnc_mode = rs.rs400_advanced_mode(find_device_that_supports_advanced_mode())
depth_table = advnc_mode.get_depth_table()
depth_table.depthClampMax = 400;  # 1m30 if depth unit at 0.001
depth_table.depthClampMin = 10;  #
advnc_mode.set_depth_table(depth_table)


pc = rs.pointcloud()
points = rs.points()

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
profile = pipeline.get_active_profile()

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()


clipping_distance = 2000 #(millimeters)
align = rs.align(rs.stream.color)

colorizer = rs.colorizer()

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    colorized = colorizer.process(frames)


    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = aligned_depth_frame.get_data()
    color_image = color_frame.get_data()

    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 153, color_image)
    # print(type(depth_image_3d))

    points = pc.calculate(aligned_depth_frame)

    cv2.imshow('Bg removed', bg_removed)
    key = cv2.waitKey(1)
    if key == ord("e"):
        points.export_to_ply('./outp32ut_1.ply', aligned_depth_frame)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

pipeline.stop()