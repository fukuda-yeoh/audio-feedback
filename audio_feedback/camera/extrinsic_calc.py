import json

import cv2 as cv
import numpy as np


def load_extrinsic(file_path):
    with open(file_path, mode="r") as f:
        cal_data = json.load(f)
        if cal_data["corners_in"] is not None:
            corners_in = np.array(cal_data["corners_in"]).astype(np.int_)
            corners_out = np.array(cal_data["corners_out"]).astype(np.int_)
            output_size = np.array(cal_data["output_size"]).astype(np.int_)

            transformation_matrix, mask = cv.findHomography(
                corners_in, corners_out, cv.RANSAC
            )

            return transformation_matrix, mask, output_size


def change_perspective(img, transformation_matrix, output_size):
    return cv.warpPerspective(img, transformation_matrix, output_size)


if __name__ == "__main__":
    my_img = cv.imread("../../img_files/sample.jpg")

    c_in = [
        [255, 512],
        [255, 121],
        [979, 136],
        [969, 519],
    ]
    c_out = [
        [0, 0],
        [0, 100],
        [200, 100],
        [200, 0],
    ]
    out_size = (200, 100)

    trans_mat, _ = cv.findHomography(np.array(c_in), np.array(c_out), cv.RANSAC)
    out_img = change_perspective(my_img, trans_mat, out_size)

    cv.imshow(f"In", my_img)
    cv.imshow(f"Out", out_img)

    cv.waitKey()
