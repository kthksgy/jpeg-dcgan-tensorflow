import cv2
import numpy as np


class YCrCb:
    '''画像をRGBからYCrCbに変換するクラス
    '''
    def __call__(self, image: np.ndarray) -> np.ndarray:
        '''入力画像をRGBからYCrCbに変換する。
        Args:
            image: RGBの画像
        Returns:
            YCrCbの画像
        Note:
            http://opencv.jp/opencv-2.1/cpp/miscellaneous_image_transformations.html
        '''
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            return image
