import cv2
import numpy as np


class YCrCb:
    '''画像をBGRからYCrCbに変換するクラス
    '''
    def __call__(self, image: np.ndarray) -> np.ndarray:
        '''入力画像をBGRからYCrCbに変換する。
        Args:
            image: BGRの画像
        Returns:
            YCrCbの画像
        Note:
            http://opencv.jp/opencv-2.1/cpp/miscellaneous_image_transformations.html
        '''
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            return image

    def inverse(self, image: np.ndarray) -> np.ndarray:
        '''入力画像をYCrCbからBGRに変換する。
        Args:
            image: YCrCbの画像
        Returns:
            BGRの画像
        '''
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        else:
            return np.tile(image, (1, 1, 3))
