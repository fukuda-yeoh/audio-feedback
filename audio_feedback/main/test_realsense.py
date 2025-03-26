import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# HSV color range for orange (example for an orange object)
lower_orange = np.array([6, 140, 150])
upper_orange = np.array([20, 220, 250])

try:
    while True:
        # Wait for frames from the camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert BGR to HSV for color detection
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Create a mask for the orange color
        mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

        # Find contours of the masked area (i.e., the orange object)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assuming it's the object we want)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Draw a rectangle around the detected object
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw a circle at the center of the detected object
            cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)

            # Get depth values for 100 points around the center (10*10 grid)
            # points = [
            #     (center_x - 1, center_y - 1),
            #     (center_x, center_y - 1),
            #     (center_x + 1, center_y - 1),
            #     (center_x - 1, center_y),
            #     (center_x, center_y),
            #     (center_x + 1, center_y),
            #     (center_x - 1, center_y + 1),
            #     (center_x, center_y + 1),
            #     (center_x + 1, center_y + 1),
            # ]

            # List to hold valid depth values
            # valid_depths = []

            # for px, py in points:
            #     if 0 <= px < depth_image.shape[1] and 0 <= py < depth_image.shape[0]:
            #         depth_value = depth_image[py, px]
            # if depth_value != 0:  # Exclude points with no valid depth
            #             valid_depths.append(depth_value * depth_frame.get_units())

            window_size = 5
            window = (
                depth_image[
                    int(center_y - window_size / 2) : int(center_y + window_size / 2),
                    int(center_x - window_size / 2) : int(center_x + window_size / 2),
                ]
                * depth_frame.get_units()
            )
            window[window == 0] = np.nan

            # Calculate the median depth from valid points
            median_depth = np.nanmedian(window)

            # Display the median depth on the image
            cv2.putText(
                color_image,
                f"Median Distance: {median_depth:.3f} m",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        # Display the images
        cv2.imshow("Color Image with Object", color_image)
        cv2.imshow("Mask", mask)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
