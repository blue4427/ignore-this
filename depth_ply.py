import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import time

ser = serial.Serial('COM3', 9600)
time.sleep(2)

pc = rs.pointcloud()
points = rs.points()

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


pipeline.start(config)

clipping_distance = 500 #(millimeters)
align = rs.align(rs.stream.color)

colorizer = rs.colorizer()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        colorized = colorizer.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = aligned_depth_frame.get_data()
        color_image = color_frame.get_data()


        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 153, color_image)
        #print(type(depth_image_3d))

        points = pc.calculate(aligned_depth_frame)

        cv2.imshow('Bg removed', bg_removed)
        key = cv2.waitKey(1)
        if key == ord("e"):
            points.export_to_ply('./output_1.ply', aligned_depth_frame)
            ser.write(b'1')
            #time.sleep(2)
            points.export_to_ply('./output_2.ply', aligned_depth_frame)
            ser.write(b'2')
            #time.sleep(2)
            points.export_to_ply('./output_3.ply', aligned_depth_frame)
            ser.write(b'3')
            #time.sleep(2)
            points.export_to_ply('./output_4.ply', aligned_depth_frame)
            ser.write(b'4')
            #time.sleep(2)
            ser.write(b'5')
            ser.close()
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()