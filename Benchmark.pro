QT -= gui console

CONFIG += c++11
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp

LIBS += -L/usr/local/lib        \
        -lopencv_calib3d        \
        -lopencv_core           \
        #-lopencv_dnn            \
        #-lopencv_flann          \
        #-lopencv_ml             \
        #-lopencv_shape          \
        #-lopencv_stitching      \
        #-lopencv_superres       \
        -lopencv_features2d     \
        -lopencv_highgui        \
        -lopencv_imgcodecs      \
        -lopencv_imgproc        \
        -lopencv_objdetect      \
        -lopencv_photo          \
        -lopencv_video          \
        -lopencv_videoio        \
        -lopencv_videostab      \
# opencv with cuda libs
        -lopencv_cudaarithm         \
        -lopencv_cudabgsegm         \
        -lopencv_cudacodec          \
        -lopencv_cudafeatures2d     \
        -lopencv_cudafilters        \
        -lopencv_cudaimgproc        \
        -lopencv_cudalegacy         \
        -lopencv_cudaobjdetect      \
        -lopencv_cudaoptflow        \
        -lopencv_cudastereo         \
        -lopencv_cudawarping        \
        -lopencv_cudev              \
# cuda libs
        -lcudart

HEADERS += \
    Tools.hpp

DEFINES += TEST_IMAGES_PATH=\\\"$$PWD\\\"

INCLUDEPATH += /usr/local/cuda/include/
