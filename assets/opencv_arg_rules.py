OPENCV_ARG_RULES = {
    "bilateralFilter": {
        "args": {
            1: {"type": "int", "constraints": {"min": 1, "odd": True}},
            2: {"type": "float", "constraints": {"min": 0.0}},
            3: {"type": "float", "constraints": {"min": 0.0}},
            5: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "blur": {
        "args": {
            1: {"type": "tuple[int,int]", "constraints": {"odd": True}},
            4: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "boxFilter": {
        "args": {
            1: {"type": "int", "constraints": {}},
            2: {"type": "tuple[int,int]", "constraints": {"odd": True}},
            5: {"type": "bool", "constraints": {}},
            6: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "dilate": {
        "args": {
            3: {"type": "int", "constraints": {"min": 1}},
            4: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "erode": {
        "args": {
            3: {"type": "int", "constraints": {"min": 1}},
            4: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "GaussianBlur": {
        "args": {
            1: {"type": "tuple[int,int]", "constraints": {"odd": True}},
            2: {"type": "float", "constraints": {}},
            4: {"type": "float", "constraints": {}},
            5: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "getDerivKernels": {
        "args": {
            0: {"type": "int", "constraints": {}},
            1: {"type": "int", "constraints": {}},
            2: {"type": "int", "constraints": {"odd": True}},
            5: {"type": "bool", "constraints": {}},
            6: {"type": "int", "constraints": {}}
        }
    },
    "getGaborKernel": {
        "args": {
            0: {"type": "tuple[int,int]", "constraints": {"odd": True}},
            1: {"type": "float", "constraints": {}},
            2: {"type": "float", "constraints": {}},
            3: {"type": "float", "constraints": {}},
            4: {"type": "float", "constraints": {}},
            5: {"type": "float", "constraints": {}}
        }
    },
    "getGaussianKernel": {
        "args": {
            0: {"type": "int", "constraints": {"odd": True}},
            1: {"type": "float", "constraints": {}}
        }
    },
    "getStructuringElement": {
        "args": {
            0: {"type": "int", "constraints": {"enum": [0, 1, 2]}},
            1: {"type": "tuple[int,int]", "constraints": {"odd": True}}
        }
    },
    "Laplacian": {
        "args": {
            1: {"type": "int", "constraints": {}},
            3: {"type": "int", "constraints": {"odd": True}},
            4: {"type": "float", "constraints": {}},
            5: {"type": "float", "constraints": {}},
            6: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "medianBlur": {
        "args": {
            1: {"type": "int", "constraints": {"odd": True}}
        }
    },
    "pyrDown": {
        "args": {
            3: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "pyrMeanShiftFiltering": {
        "args": {
            1: {"type": "float", "constraints": {}},
            2: {"type": "float", "constraints": {}},
            4: {"type": "int", "constraints": {"min": 0}}
        }
    },
    "pyrUp": {
        "args": {
            3: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "Scharr": {
        "args": {
            1: {"type": "int", "constraints": {}},
            5: {"type": "float", "constraints": {}},
            6: {"type": "float", "constraints": {}},
            7: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "Sobel": {
        "args": {
            1: {"type": "int", "constraints": {}},
            5: {"type": "int", "constraints": {"odd": True, "min": 3}},
            6: {"type": "float", "constraints": {}},
            7: {"type": "float", "constraints": {}},
            8: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "morphologyEx": {
        "args": {
            1: {
                "type": "int",
                "constraints": {
                    "enum": [
                        2,  # cv.MORPH_ERODE
                        3,  # cv.MORPH_DILATE
                        4,  # cv.MORPH_OPEN
                        5,  # cv.MORPH_CLOSE
                        6,  # cv.MORPH_GRADIENT
                        7,  # cv.MORPH_TOPHAT
                        8,  # cv.MORPH_BLACKHAT
                        9   # cv.MORPH_HITMISS
                    ]
                }
            },
            4: {"type": "tuple[int, int]", "constraints": {}},  # anchor point
            5: {"type": "int", "constraints": {"min": 1}},      # iterations
            6: {
                "type": "int",
                "constraints": {
                    "enum": [
                        0,  # cv.BORDER_CONSTANT
                        1,  # cv.BORDER_REPLICATE
                        2,  # cv.BORDER_REFLECT
                        3,  # cv.BORDER_WRAP
                        4   # cv.BORDER_REFLECT_101 or cv.BORDER_DEFAULT
                    ]
                }
            },
        }
    },
    "spatialGradient": {
        "args": {
            2: {"type": "int", "constraints": {"odd": True}},
            3: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "sqrBoxFilter": {
        "args": {
            1: {"type": "int", "constraints": {}},
            2: {"type": "tuple[int,int]", "constraints": {"odd": True}},
            4: {"type": "bool", "constraints": {}},
            5: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}
        }
    },
    "stackBlur": {
        "args": {
            1: {"type": "int", "constraints": {"odd": True}}
        }
    },
    "Canny": {
        "args": {
            1: {"type": "float", "constraints": {"min": 0.0}},    # threshold1
            2: {"type": "float", "constraints": {"min": 0.0}},    # threshold2
            4: {"type": "int", "constraints": {"odd": True, "min": 3}},  # apertureSize
            5: {"type": "bool", "constraints": {}}               # L2gradient
        }
    },
    "cornerEigenValsAndVecs": {
        "args": {
            1: {"type": "int", "constraints": {"min": 1}},       # blockSize
            2: {"type": "int", "constraints": {"odd": True, "min": 3}},  # ksize
            4: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}  # borderType
        }
    },
    "cornerHarris": {
        "args": {
            1: {"type": "int", "constraints": {"min": 1}},       # blockSize
            2: {"type": "int", "constraints": {"odd": True, "min": 3}},  # ksize
            3: {"type": "float", "constraints": {"min": 0.0}},    # k
            5: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}  # borderType
        }
    },
    "cornerMinEigenVal": {
        "args": {
            1: {"type": "int", "constraints": {"min": 1}},       # blockSize
            3: {"type": "int", "constraints": {"odd": True, "min": 3}},  # ksize
            4: {"type": "int", "constraints": {"enum": [0, 1, 2, 3, 4]}}  # borderType
        }
    },
    "goodFeaturesToTrack": {
        "args": {
            1: {"type": "int", "constraints": {"min": 1}},            # maxCorners
            2: {"type": "float", "constraints": {"min": 0.0, "max": 1.0}},  # qualityLevel
            3: {"type": "float", "constraints": {"min": 0.0}},        # minDistance
            6: {"type": "int", "constraints": {"min": 1}},            # blockSize
            7: {"type": "bool", "constraints": {}},                   # useHarrisDetector
            8: {"type": "float", "constraints": {"min": 0.0}}         # k (Harris detector free parameter)
        }
    },
    "HoughCircles": {
        "args": {
            1: {"type": "int", "constraints": {"enum": [1, 2]}},      # method, usually cv.HOUGH_GRADIENT → 3 or cv.HOUGH_GRADIENT_ALT → 4, but often passed as 1/2
            2: {"type": "float", "constraints": {"min": 1.0}},        # dp, inverse accumulator resolution
            3: {"type": "float", "constraints": {"min": 1.0}},        # minDist between centers
            5: {"type": "float", "constraints": {"min": 0.0}},        # param1, higher threshold for Canny
            6: {"type": "float", "constraints": {"min": 0.0}},        # param2, accumulator threshold
            7: {"type": "int", "constraints": {"min": 0}},            # minRadius
            8: {"type": "int", "constraints": {"min": 0}}             # maxRadius
        }
    },
    "HoughLines": {
        "args": {
            1: {"type": "float", "constraints": {"min": 0.0}},        # rho
            2: {"type": "float", "constraints": {"min": 0.0}},        # theta
            3: {"type": "int", "constraints": {"min": 1}},            # threshold
            5: {"type": "float", "constraints": {"min": 0.0}},        # srn (for multi-scale Hough), usually 0
            6: {"type": "float", "constraints": {"min": 0.0}},        # stn (for multi-scale Hough), usually 0
            7: {"type": "float", "constraints": {"min": 0.0}},        # min_theta
            8: {"type": "float", "constraints": {"min": 0.0}}         # max_theta
        }
    },
    "HoughLinesP": {
        "args": {
            1: {"type": "float", "constraints": {"min": 0.0}},        # rho
            2: {"type": "float", "constraints": {"min": 0.0}},        # theta
            3: {"type": "int", "constraints": {"min": 1}},            # threshold
            5: {"type": "float", "constraints": {"min": 0.0}},        # minLineLength
            6: {"type": "float", "constraints": {"min": 0.0}}         # maxLineGap
        }
    },
    "HoughLinesPointSet": {
        "args": {
            1: {"type": "int", "constraints": {"min": 1}},            # lines_max
            2: {"type": "int", "constraints": {"min": 1}},            # threshold
            3: {"type": "float", "constraints": {"min": 0.0}},        # min_rho
            4: {"type": "float", "constraints": {"min": 0.0}},        # max_rho
            5: {"type": "float", "constraints": {"min": 0.0}},        # rho_step
            6: {"type": "float", "constraints": {"min": 0.0}},        # min_theta
            7: {"type": "float", "constraints": {"min": 0.0}},        # max_theta
            8: {"type": "float", "constraints": {"min": 0.0}}         # theta_step
        }
    },
    "preCornerDetect": {
        "args": {
            1: {"type": "int", "constraints": {"odd": True, "min": 3}},  # ksize
            3: {"type": "int", "constraints": {"enum": [0,1,2,3,4]}}    # borderType
        }
    },
    "calcBackProject": {
        "args": {
            5: {"type": "float", "constraints": {"min": 0.0}},         # scale
            6: {"type": "bool", "constraints": {}}                     # uniform
        }
    },
    "calcHist": {
        "args": {
            6: {"type": "bool", "constraints": {}},                    # accumulate
            7: {"type": "bool", "constraints": {}}                    # uniform
        }
    },
    "compareHist": {
        "args": {
            3: {"type": "int", "constraints": {"enum": [0,1,2,3,4]}},  # method
        }
    },
    "createCLAHE": {
        "args": {
            1: {"type": "float", "constraints": {"min": 0.0}},         # clipLimit
            2: {"type": "tuple[int,int]", "constraints": {"min":1}},   # tileGridSize
        }
    },
    "addWeighted": {
        "args": {
            1: {"type": "float", "constraints": {}},                   # alpha
            3: {"type": "float", "constraints": {}},                   # beta
            4: {"type": "float", "constraints": {}},                   # gamma
            6: {"type": "int", "constraints": {}}                     # dtype
        }
    },
    "adaptiveThreshold": {
        "args": {
            1: {"type": "float", "constraints": {}},                   # maxValue
            2: {"type": "int", "constraints": {"enum": [0,1]}},        # adaptiveMethod
            3: {"type": "int", "constraints": {"enum": [0,1,2,3]}},    # thresholdType
            4: {"type": "int", "constraints": {"odd": True, "min":3}}, # blockSize
            5: {"type": "float", "constraints": {}}                   # C
        }
    },
    "distanceTransform": {
        "args": {
            1: {"type": "int", "constraints": {"enum": [1,2,3]}},      # distanceType
            2: {"type": "int", "constraints": {"odd": True, "min":3}}, # maskSize
            4: {"type": "int", "constraints": {}}                     # dstType
        }
    },
    "integral": {
        "args": {
            2: {"type": "int", "constraints": {}}                     # sdepth
        }
    },
    "integral2": {
        "args": {
            3: {"type": "int", "constraints": {}} ,                   # sdepth
            4: {"type": "int", "constraints": {}}                    # sqdepth
        }
    },
    "integral3": {
        "args": {
            4: {"type": "int", "constraints": {}},                   # sdepth
            5: {"type": "int", "constraints": {}}                    # sqdepth
        }
    },
    "threshold": {
        "args": {
            1: {"type": "float", "constraints": {}},                  # thresh
            2: {"type": "float", "constraints": {}},                  # maxval
            3: {"type": "int", "constraints": {"enum": [0,1,2,3,4]}},# type
        }
    },
    "fastNlMeansDenoising": {
        "args": {
            2: {"type": "float", "constraints": {"min":0.0}},         # h
            3: {"type": "int", "constraints": {"odd":True, "min":1}}, # templateWindowSize
            4: {"type": "int", "constraints": {"odd":True, "min":1}}, # searchWindowSize
        }
    },
    "fastNlMeansDenoisingColored": {
        "args": {
            2: {"type": "float", "constraints": {"min":0.0}},         # h
            3: {"type": "float", "constraints": {"min":0.0}},         # hColor
            4: {"type": "int", "constraints": {"odd":True, "min":1}}, # templateWindowSize
            5: {"type": "int", "constraints": {"odd":True, "min":1}}, # searchWindowSize
        }
    },


}