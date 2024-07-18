import cv2 as cv
import numpy as np


class HSVColorModel:
    def __init__(self, hue_range, saturation_range, value_range):
        self.h_lower, self.h_upper = (round(x) for x in hue_range)
        self.s_lower, self.s_upper = (round(x) for x in saturation_range)
        self.v_lower, self.v_upper = (round(x) for x in value_range)

    def predict(self, img):
        thresh_img = self.in_range(img)
        x, w, y, h = self.bounding_rect(thresh_img)

        return HSVColorModelResult(img, thresh_img, x, w, y, h)

    def in_range(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        return cv.inRange(
            hsv,
            np.array([self.h_lower, self.s_lower, self.v_lower]),
            np.array([self.h_upper, self.s_upper, self.v_upper]),
        )

    @staticmethod
    def bounding_rect(bin_img):
        contours, hierarchy = cv.findContours(bin_img, 1, 2)
        if len(contours) > 0:
            cnt = contours[0]
            x, y, w, h = cv.boundingRect(cnt)
            return x, y, w, h
        else:
            return None, None, None, None


class HSVColorModelResult:
    def __init__(self, img, thresh_img, x, y, w, h):
        self.img = img
        self.thresh_img = thresh_img
        self.pos = x, y, w, h
        if x is None:
            self.detected = False
            self.annotated_img = img
        else:
            self.detected = True
            self.annotated_img = self.plot(img, x, y, w, h)
            self.center = (x + w / 2, y + h / 2)

    @staticmethod
    def plot(img, x, y, w, h):
        img = img.copy()
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

    def __bool__(self):
        return self.detected


if __name__ == "__main__":
    my_img = cv.imread("../../img_files/sample.jpg")

    model = HSVColorModel(
        hue_range=(100, 115), saturation_range=(25, 255), value_range=(150, 255)
    )
    result = model.predict(my_img)

    cv.imshow(f"Raw", my_img)
    cv.imshow(f"In Range", result.thresh_img)
    cv.imshow(f"Recognition", result.annotated_img)

    cv.waitKey()
