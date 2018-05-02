from utils import *
import m_utils
from line import Line
import os
import time
from moviepy.editor import *


def run(file):
    # 如果相机纠正数据和透视数据文件不存在则计算数据并记录否则加载数据
    if not os.path.exists('temp.pkl'):
        mtx, dist, src, dst = m_utils.store_data([640, 128])
    else:
        mtx, dist, src, dst = m_utils.load_data()
    camera = cv2.VideoCapture(file)
    left_line = Line()
    right_line = Line()
    out = cv2.VideoWriter('video/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          10, (640, 680))
    
    while True:
        _, image = camera.read()
        # show_img(cv2.resize(image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA),10000,'image')

        # 相机失真的矫正
        undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
        # show_img(undistorted_image,10000,'undistorted')

        # 图像缩小一倍
        zoom_image = cv2.resize(undistorted_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        # show_img(zoom_image, 100000, 'zoom')

        # 高斯模糊
        # gaussian_image = m_utils.gaussian_blur(zoom_image)
        # show_img(gaussian_image, 10000, 'gaussian')

        # 截取出roi区域（图片中含有道路）
        shape = undistorted_image.shape
        roi_image = zoom_image[220:shape[0] - 12, 0:shape[1], 2]
        roi_color_image = zoom_image[220:shape[0] - 12, 0:shape[1]]
        # show_img(roi_color_image, 100000, 'roi')

        # ----------梯度下降------------
        # 以红色通道进行梯度下降
        gradient_image = m_utils.gradient_combine(roi_image,
                                                  (35, 100),
                                                  (30, 255),
                                                  (30, 255),
                                                  (0.7, 1.3))
        # show_img(gradient_image, 10000, 'gradient')
        # hls色彩空间梯度下降
        hls_image = m_utils.hls_combine(roi_color_image, (10, 100), (0, 60), (85, 255))
        # show_img(hls_image, 10000, 'hls')

        # 将hls和梯度下降得到的图像再次合并
        result = np.zeros_like(gradient_image).astype(np.uint8)
        result[(gradient_image > 1)] = 100
        result[(hls_image > 1)] = 255
        # show_img(result, 10000, 'result')

        # import pylab
        # pylab.imshow(result)
        # pylab.plot([110, 296, 530, 344], [180, 5, 180, 5], 'r*')
        # # pylab.show()

        # 透视图像使图像呈俯视角度
        perspective_image, M, Minv = warp_image(result, src=src, dst=dst, size=(720, 720))
        # perspective_image_color=[]
        # time_=time.time()
        # for row in perspective_image:
        #     temp=[]
        #     for point in row:
        #         temp.append([point,point,point])
        #     perspective_image_color.append(temp)
        # print(time.time()-time_)
        perspective_image_color=np.dstack((perspective_image,perspective_image,perspective_image))

        # zoom_perspective_image = cv2.resize(perspective_image, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
        # show_img(perspective_image, 10000, 'perspective')

        # 寻找车道线
        try:
            search_image = find_lr_lines(perspective_image, left_line=left_line, right_line=right_line)
        except Exception as e:
            left_line.detected = False
            print(e)

        # show_img(search_image,100000,'search')
        # 绘制车道线
        w_color_result = draw_lane(search_image, left_line, right_line)
        # show_img(w_comb_result, 10000, 'w_comb')
        # show_img(w_color_result, 10000, 'w_color')

        # 将绘制车道线的俯视角度透视成原角度
        color_road = cv2.warpPerspective(w_color_result, Minv, (result.shape[1], result.shape[0]))
        # show_img(color_road, 10000, 'color_road')

        # 将透视回原角度的车道线标注叠加到原图像中
        mask = np.zeros_like(zoom_image)
        mask[220:shape[0] - 12, 0:shape[1]] = color_road
        road_image = cv2.addWeighted(zoom_image, 1, mask, 0.3, 0)
        # show_img(road_image, 10000, 'result')

        # 打印车辆位置信息
        info_road = m_utils.print_vehicle_data(image=road_image, left_line=left_line, right_line=right_line)

        # Parallel visualization
        debug_image=np.hstack((cv2.resize(np.array(perspective_image_color),(320,320)),cv2.resize(search_image,(320,320))))
        # show_img(debug_image,10000,'debug')
        debug_image=np.vstack((debug_image,info_road))

        # show_img(debug_image,10,'road')
        # out.write(debug_image)
    out.release()
    camera.release()

def process_image(image):
    # 相机失真的矫正
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    # show_img(undistorted_image,10000,'undistorted')

    # 图像缩小一倍
    zoom_image = cv2.resize(undistorted_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    # show_img(zoom_image, 100000, 'zoom')

    # 高斯模糊
    # gaussian_image = m_utils.gaussian_blur(zoom_image)
    # show_img(gaussian_image, 10000, 'gaussian')

    # 截取出roi区域（图片中含有道路）
    shape = undistorted_image.shape
    roi_image = zoom_image[220:shape[0] - 12, 0:shape[1], 2]
    roi_color_image = zoom_image[220:shape[0] - 12, 0:shape[1]]
    # show_img(roi_color_image, 100000, 'roi')

    # ----------梯度下降------------
    # 以红色通道进行梯度下降
    gradient_image = m_utils.gradient_combine(roi_image,
                                              (35, 100),
                                              (30, 255),
                                              (30, 255),
                                              (0.7, 1.3))
    # show_img(gradient_image, 10000, 'gradient')
    # hls色彩空间梯度下降
    hls_image = m_utils.hls_combine(roi_color_image, (10, 100), (0, 60), (85, 255))
    # show_img(hls_image, 10000, 'hls')

    # 将hls和梯度下降得到的图像再次合并
    result = np.zeros_like(gradient_image).astype(np.uint8)
    result[(gradient_image > 1)] = 100
    result[(hls_image > 1)] = 255
    # show_img(result, 10000, 'result')

    # import pylab
    # pylab.imshow(result)
    # pylab.plot([110, 296, 530, 344], [180, 5, 180, 5], 'r*')
    # # pylab.show()

    # 透视图像使图像呈俯视角度
    perspective_image, M, Minv = warp_image(result, src=src, dst=dst, size=(720, 720))
    # perspective_image_color=[]
    # time_=time.time()
    # for row in perspective_image:
    #     temp=[]
    #     for point in row:
    #         temp.append([point,point,point])
    #     perspective_image_color.append(temp)
    # print(time.time()-time_)
    perspective_image_color = np.dstack((perspective_image, perspective_image, perspective_image))

    # zoom_perspective_image = cv2.resize(perspective_image, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
    # show_img(perspective_image, 10000, 'perspective')

    # 寻找车道线
    try:
        search_image = find_lr_lines(perspective_image, left_line=left_line, right_line=right_line)
    except Exception as e:
        left_line.detected = False
        print(e)

    # show_img(search_image,100000,'search')
    # 绘制车道线
    w_color_result = draw_lane(search_image, left_line, right_line)
    # show_img(w_comb_result, 10000, 'w_comb')
    # show_img(w_color_result, 10000, 'w_color')

    # 将绘制车道线的俯视角度透视成原角度
    color_road = cv2.warpPerspective(w_color_result, Minv, (result.shape[1], result.shape[0]))
    # show_img(color_road, 10000, 'color_road')

    # 将透视回原角度的车道线标注叠加到原图像中
    mask = np.zeros_like(zoom_image)
    mask[220:shape[0] - 12, 0:shape[1]] = color_road
    road_image = cv2.addWeighted(zoom_image, 1, mask, 0.3, 0)
    # show_img(road_image, 10000, 'result')

    # 打印车辆位置信息
    info_road = m_utils.print_vehicle_data(image=road_image, left_line=left_line, right_line=right_line)

    # Parallel visualization
    debug_image = np.hstack(
        (cv2.resize(np.array(perspective_image_color), (320, 320)), cv2.resize(search_image, (320, 320))))
    # show_img(debug_image,10000,'debug')
    debug_image = np.vstack((debug_image, info_road))

    return debug_image

if __name__ == '__main__':
    # 如果相机纠正数据和透视数据文件不存在则计算数据并记录否则加载数据
    if not os.path.exists('temp.pkl'):
        mtx, dist, src, dst = m_utils.store_data([640, 128])
    else:
        mtx, dist, src, dst = m_utils.load_data()
    left_line = Line()
    right_line = Line()
    # run('project_video.mp4')
    white_output = 'video/output.mp4'
    clip1 = VideoFileClip("video/project_video.mp4")
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
