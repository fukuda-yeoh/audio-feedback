# Libraries
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time # timeモジュールを追加

def findDevices():
    ctx = rs.context()  # Create librealsense context for managing devices
    serials = []
    if len(ctx.devices) > 0:
        for dev in ctx.devices:
            print(
                "Found device: ",
                dev.get_info(rs.camera_info.name),
                " ",
                dev.get_info(rs.camera_info.serial_number),
            )
            serials.append(dev.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")

    return serials, ctx


def setupMasterSlave(serials, ctx):
    if len(serials) < 2:
        print("At least two devices are required for master-slave setup.")
        # この場合、Falseを返すのではなく、単に処理を終了させます。
        # メインループでこの関数の戻り値が使われていないため、ここでは特に問題ありません。
        return

    # Assume the first device is the master and the rest are slaves
    for i, device in enumerate(ctx.devices):
        print(
            f"Configuring device {i+1} with serial {device.get_info(rs.camera_info.serial_number)}"
        )
        if i == 0:
            # Set the first device as master
            print(
                f"Setting device {device.get_info(rs.camera_info.serial_number)} as master."
            )
            device.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 1)
        else:
            # Set the rest of the devices as slaves
            print(
                f"Setting device {device.get_info(rs.camera_info.serial_number)} as slave."
            )
            device.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 2)


def enableDevices(
    serials, ctx, resolution_width=640, resolution_height=480, frame_rate=30
):
    pipelines = []
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        # depthストリームを無効化
        # cfg.enable_stream(
        #     rs.stream.depth,
        #     resolution_width,
        #     resolution_height,
        #     rs.format.z16,
        #     frame_rate,
        # )
        # カラー画像のみを有効化
        cfg.enable_stream(
            rs.stream.color,
            resolution_width,
            resolution_height,
            rs.format.bgr8,
            frame_rate,
        )
        pipe.start(cfg)
        pipelines.append([serial, pipe])

    # カメラ起動のための短い待機時間 (以前のデバッグで追加した場合、残しておくことを推奨)
    time.sleep(1)

    return pipelines


def Visualize(pipelines):
    # カラー画像のみの場合、アラインメントは不要になります
    # align_to = rs.stream.color
    # align = rs.align(align_to)

    for device, pipe in pipelines:
        # Get frameset (今回はカラーフレームのみ取得)
        # タイムアウトは必要に応じて調整
        frames = pipe.wait_for_frames(5000) # デフォルトの5秒に戻すか、必要に応じて調整

        # カラーフレームのみを取得
        color_frame = frames.get_color_frame()

        # Validate that color frame is valid
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Render images - カラー画像のみを表示
        cv.imshow("RealSense " + device, color_image) # ウィンドウ名を変更して区別しやすくすると良いでしょう
        key = cv.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:
            cv.destroyAllWindows()
            return True

        # Save images (カラー画像のみ保存)
        if key == 115: # 's'キー
            cv.imwrite(str(device) + "_color.png", color_image)
            print(f"Saved color image for device {device}")


def pipelineStop(pipelines):
    for device, pipe in pipelines:
        # Stop streaming
        pipe.stop()


# -------Main program--------

serials, ctx = findDevices()
# 複数のカメラが検出されなかった場合は、setupMasterSlaveはスキップされますが、
# シリアル番号が1つでもenableDevicesは実行されます。
# 同期は2台以上でなければ意味がないため、ここで早期終了させることも検討できます。
if len(serials) < 2:
    print("WARNING: Less than two devices found. Master-slave synchronization will not be effective.")
    # 必要であればここでsys.exit()などを呼び出してプログラムを終了させることもできます。
    # import sys
    # sys.exit()

setupMasterSlave(serials, ctx) # 2台以上のカメラが見つかれば同期設定を試みる

# Define some constants
resolution_width = 640  # pixels
resolution_height = 480  # pixels
frame_rate = 30  # fps

pipelines = enableDevices(serials, ctx, resolution_width, resolution_height, frame_rate)

try:
    while True:
        exit = Visualize(pipelines)
        if exit == True:
            print("Program closing...")
            break
finally:
    pipelineStop(pipelines)