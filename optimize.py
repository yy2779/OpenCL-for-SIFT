#!/usr/bin/env python

import time
import pyopencl as cl
import pyopencl.array
import numpy as np
import cv2

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()
        for dev in devs:
            MAX_WORKGROUP_SIZE = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

# Set up a command queue; we need to enable profiling to time GPU operations:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# read kernel from cpp file
kernel = open('test.cpp').read()
# build the kernel
prg = cl.Program(ctx, kernel).build()
mf = cl.mem_flags

# read initial image
img = cv2.imread('lenel.png', cv2.IMREAD_GRAYSCALE)  # img is a 2D numpy array

# some parameters from opencv.sift
descr_width = 4
descr_hist_bins = 8
init_sigma = 0.5
img_border = 5
max_interp_step = 5
ori_hist_bins = 36
ori_sig_fctr = 1.5
ori_radius = ori_sig_fctr * 3
ori_peak_ratio = 0.8
descr_scl_fctr = 3.0
descr_mag_thr = 0.2
int_descr_fctr = 512
nfeatures = 0
sigma = 1.6
nOctaveLayers = 3
contrastThreshold = 0.025
edgeThreshold = 1.4
flt_epsilon = 1.19209290E-07
firstOctave = -1


print("  1. build Gaussian Pyramid: ")
# implemented by yl3747
class sift(object):
    def __init__(self):
        if ((int)(np.sqrt(MAX_WORKGROUP_SIZE)) == np.sqrt(MAX_WORKGROUP_SIZE)):
            self.LOCAL_X = (int)(np.sqrt(MAX_WORKGROUP_SIZE))
            self.LOCAL_Y = (int)(np.sqrt(MAX_WORKGROUP_SIZE))
        else:
            self.LOCAL_X = (int)(np.sqrt(MAX_WORKGROUP_SIZE / 2))
            self.LOCAL_Y = (int)(np.sqrt(MAX_WORKGROUP_SIZE / 2) * 2)

        self.intsz = (np.int32(1)).itemsize
        self.floatsz = (np.float32(1.0)).itemsize
        self.intsz16 = (np.int16(1)).itemsize
        self.floatsz16 = (np.float16(1.0)).itemsize

    def createInitalImage(self, img):
        # called by the __call__ function
        # calculate the base image for GaussianPyramid
        # the img should be a 2D matrix, which is a gray image already
        sig_diff = np.sqrt(np.maximum(sigma ** 2 - 4 * init_sigma ** 2, 0.01)).astype(np.float32)
        size = np.round(sig_diff * 6 + 1).astype(np.int32)
        if size % 2 == 0:
            size += 1

        dbl = cv2.resize(img, (2 * img.shape[0], 2 * img.shape[1])).astype(np.uint32)
        dst = np.zeros_like(dbl).astype(np.uint32)  # output
        dbl_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dbl)
        dst_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=dst)

        local_src_x = self.LOCAL_X + size - 1
        local_src_y = self.LOCAL_Y + size - 1
        event = prg.GaussianBlur(queue, dst.shape, (self.LOCAL_X, self.LOCAL_Y), dbl_buf, dst_buf,
                                 cl.LocalMemory(local_src_x * local_src_y * self.intsz), np.int32(local_src_x),
                                 np.int32(local_src_y), cl.LocalMemory(size * size * self.intsz),
                                 np.float32(sig_diff), cl.LocalMemory(self.intsz), np.uint32(dbl.shape[0]),
                                 np.uint32(dbl.shape[1]))
        event.wait()
        print('GPU GaussianBlur time: {}'.format((1e-9 * (event.profile.end - event.profile.start))))
        cl.enqueue_copy(queue, dst, dst_buf)
        dbl_buf.release()
        dst_buf.release()

        start = time.time()
        base_verify = cv2.GaussianBlur(dbl.astype(np.uint8), (int(size), int(size)), sig_diff)
        end = time.time()
        print('cv2 GaussianBlur time: {}'.format(end - start))

        base_diff = np.sum((dst.astype(np.int32) - base_verify.astype(np.int32)))
        print('Test whether GPU equal to cv2:')
        print(base_diff / (dst.shape[0] * dst.shape[1]) < 1)  # if the average difference is smaller than 1

        return dst

    def buildGaussianPyramid(self, base_img):
        nOctaves = (np.round(np.log(np.minimum(base_img.shape[0], base_img.shape[1])) / np.log(2) - 2) - (-1)).astype(np.int32)
        sig = np.zeros([nOctaveLayers + 3]).astype(np.float32)  # Gaussian sigmas, shape [nOctaveLayers + 3]
        pyramid = []  # Gaussian Pyramid, should be a list with length [nOctaves * (nOctaveLayers + 3)]
        gpyr_verify = []  # used for verify the results

        sig[0] = sigma
        k = np.power(2.0, 1.0 / nOctaveLayers)

        # calculate all the sigmas
        for ol in range(1, nOctaveLayers + 3):
            sig_prev = np.power(k, (ol - 1.0)) * sigma
            sig_total = sig_prev * k
            sig[ol] = np.sqrt(sig_total ** 2 - sig_prev ** 2)
        sigma_max = sig[-1]
        max_size = np.round(sigma_max * 6 + 1).astype(np.int32)
        GPU_time = []
        CPU_time = []
        if max_size % 2 == 0:
            max_size += 1
        # calculate the Pyramid
        # NOTE: the images in different Octaves have different shape, I keep the loop among Octaves to simplify the kernel.
        for o in range(nOctaves):
            if o == 0:  # the base image of the whole Pyramid
                src = base_img
            else:  # the base image of an Octave
                src = pyramid[(o - 1) * (nOctaveLayers + 3) + nOctaveLayers]
                src = cv2.resize(src, (src.shape[0] // 2, src.shape[1] // 2)).astype(np.int32)  # error

            pyramid.append(np.uint8(src))
            gpyr_verify.append(np.uint8(src))

            dst = np.zeros([nOctaveLayers + 2, src.shape[0], src.shape[1]]).astype(np.int32)  # output

            # calculate the nOctaveLayers + 2 images in an Octave
            src_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(src))
            dst_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.uint32(dst))
            sig_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(sig[1:]))

            if src.shape[0] < self.LOCAL_X:
                local_x = src.shape[0]
            else:
                local_x = self.LOCAL_X
            if src.shape[1] < self.LOCAL_Y:
                local_y = src.shape[1]
            else:
                local_y = self.LOCAL_Y

            local_src_x = local_x + max_size - 1
            local_src_y = local_y + max_size - 1
            event2 = prg.GaussianPyramid(queue, (src.shape[0], src.shape[1], nOctaveLayers + 2), (local_x, local_y, 1),
                                         src_buf, dst_buf, np.int32(nOctaveLayers + 2), np.uint32(src.shape[0]),
                                         np.uint32(src.shape[1]), sig_buf, cl.LocalMemory(local_x * local_y * self.floatsz),
                                         cl.LocalMemory(self.intsz), cl.LocalMemory(local_src_x * local_src_y * self.intsz),
                                         np.int32(max_size), np.int32(local_src_x), np.int32(local_src_y))
            event2.wait()
            GPU_time.append(1e-9 * (event2.profile.end - event2.profile.start))

            cl.enqueue_copy(queue, dst, dst_buf)
            for i in range(1, nOctaveLayers + 3):
                pyramid.append(np.uint8(dst[i - 1]))  # append the result
            sig_buf.release()
            src_buf.release()
            dst_buf.release()

            start = time.time()
            for i in range(1, nOctaveLayers + 3):
                size_k = np.round(sig[i] * 3 * 2 + 1).astype(np.int32)
                if size_k % 2 == 0:
                    size_k += 1
                dst = cv2.GaussianBlur(src.astype(np.uint8), (int(size_k), int(size_k)), sig[i])
                gpyr_verify.append(np.uint8(dst))
            end = time.time()
            CPU_time.append(end-start)

        # print(GPU_time)
        assert len(pyramid) == nOctaves * (nOctaveLayers + 3)
        print('GPU GaussianPyramid time: {}'.format(np.sum(GPU_time)))
        print('cv2 GaussianPyramid time: {}'.format(np.sum(CPU_time)))
        pyr_diff = 0
        for o in range(nOctaves):
            for l in range(nOctaveLayers + 3):
                pyr_diff += np.sum((pyramid[o * (nOctaveLayers + 3) + l].astype(np.int32) - gpyr_verify[o * (nOctaveLayers + 3) + l].astype(np.int32)))
        print('Test whether GPU equal to cv2:')
        print(pyr_diff / (pyramid[0].shape[0] * pyramid[0].shape[1]) < 1)
        return pyramid, nOctaves

    def __call__(self, img, is_useProvidedKeypoints=False, is_descriptors_needed=True):
        base = self.createInitalImage(img)  # https://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
        pyramid, nOctaves = self.buildGaussianPyramid(base.astype(np.uint8))

        return pyramid, nOctaves


Gaussian_operation = sift()
gpyr, nOctaves = Gaussian_operation(img)


print('')
print("  2. detect keypoints: ")
# implemented by yy2779
# used to store the results
kpt_list = []
# record the kernel time
kernel_time = np.zeros([4])
test_time = 0
for o in range(nOctaves):
    gpyr_sub = np.copy(gpyr[o * (nOctaveLayers + 3):(o + 1) * (nOctaveLayers + 3)])
    rows = gpyr[o * (nOctaveLayers + 3)].shape[0]
    cols = gpyr[o * (nOctaveLayers + 3)].shape[1]
    # avoid useless loop
    if rows <= 2*img_border or cols <= 2*img_border:
        break

    print('the index of Octave: {}'.format(o))
    # built DoGPyramid
    dogpyr_sub = np.zeros([nOctaveLayers + 2, rows, cols]).astype(np.int32)
    dogpyr_buf = cl.Buffer(ctx, mf.WRITE_ONLY, dogpyr_sub.nbytes)
    gpyr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpyr_sub)
    # run the optimized kernel
    evt1 = prg.DoGPyramid(queue, [rows, cols, nOctaveLayers + 2], [8, 8, 5], gpyr_buf, dogpyr_buf, np.uint32(rows), np.uint32(cols))
    evt1.wait()
    kernel_time[0] += (1e-9 * (evt1.profile.end - evt1.profile.start))
    cl.enqueue_copy(queue, dogpyr_sub, dogpyr_buf)
    dogpyr_buf.release()
    print('-- kernel DoGPyramid finished ')
    # validate
    dogpyr_python = np.zeros_like(dogpyr_sub).astype(np.int32)
    start = time.time()
    for i in range(nOctaveLayers + 2):
        dogpyr_python[i] = cv2.subtract(gpyr_sub[i+1], gpyr_sub[i])
    end = time.time()
    test_time += end -start
    sim = (dogpyr_python == dogpyr_sub).all()
    print('validate the kernel with cv2: ')
    print(sim)

    # find local extrema
    threshold = int(0.5 * contrastThreshold / nOctaveLayers * 255)
    kpts_out1 = np.zeros([nOctaveLayers * rows * cols, 4]).astype(np.float32)  # used to record the keypoints
    idx = np.array([0]).astype(np.int32)
    idx_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
    dogpyr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dogpyr_sub)
    kpts_out1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, kpts_out1.nbytes)
    # run the optimized kernel
    evt2 = prg.findLocalExtrema(queue, [nOctaveLayers, rows, cols], [1, 16, 16], dogpyr_buf, kpts_out1_buf, idx_buf, np.uint32(o),
                                np.uint32(rows), np.uint32(cols), np.float32(threshold), np.uint32(img_border))
    evt2.wait()
    kernel_time[1] += (1e-9 * (evt2.profile.end - evt2.profile.start))
    cl.enqueue_copy(queue, kpts_out1, kpts_out1_buf)
    cl.enqueue_copy(queue, idx, idx_buf)
    idx_buf.release()
    kpts_out1_buf.release()

    kpts_result1 = np.copy(kpts_out1[0:idx[0]])
    print('-- kernel findLocalExtrema finished')
    print('the number of local extrema: {}'.format(kpts_result1.shape[0]))
    if kpts_result1.shape[0] == 0:  # didn't find local extrema
        continue

    # adjust local extrema
    kpts_out2 = np.zeros([kpts_result1.shape[0], 9]).astype(np.float32)
    idx = np.array([0]).astype(np.int32)
    idx_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
    kpts_result1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kpts_result1)  #### mf
    kpts_out2_buf = cl.Buffer(ctx, mf.WRITE_ONLY, kpts_out2.nbytes)
    local_size = 16
    global_size = int(local_size * np.ceil(kpts_result1.shape[0]/local_size))# assure the global_size is multiply of local_size
    # run the optimized kernel
    evt3 = prg.adjustLocalExtrema(queue, [global_size, 1, 1], [local_size, 1, 1], dogpyr_buf, idx_buf, kpts_out2_buf, kpts_result1_buf,
                                  np.uint32(rows), np.uint32(cols), np.uint32(nOctaveLayers), np.float32(contrastThreshold),
                                  np.float32(edgeThreshold), np.float32(sigma), np.uint32(max_interp_step),
                                  np.uint32(img_border), np.uint32(kpts_result1.shape[0]))
    evt3.wait()
    kernel_time[2] += (1e-9 * (evt3.profile.end - evt3.profile.start))
    cl.enqueue_copy(queue, kpts_out2, kpts_out2_buf)
    cl.enqueue_copy(queue, idx, idx_buf)
    idx_buf.release()
    kpts_out2_buf.release()
    dogpyr_buf.release()
    kpts_result1_buf.release()

    kpts_result2 = np.copy(kpts_out2[0:idx[0]])
    print('-- kernel adjustLocalExtrema finished')
    print('the number of adjusted keypoints: {}'.format(kpts_result2.shape[0]))
    if kpts_result2.shape[0] == 0:  # didn't find keypoints
        continue

    # calculate orientation hist
    kpts_out3 = np.zeros([2 * kpts_result2.shape[0], 6]).astype(np.float32)
    idx = np.array([0]).astype(np.int32)
    idx_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
    gpyr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gpyr_sub)
    kpts_result2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kpts_result2)
    kpts_out3_buf = cl.Buffer(ctx, mf.WRITE_ONLY, kpts_out3.nbytes)
    local_size = 256
    global_size = int(local_size * np.ceil(kpts_result2.shape[0] / local_size))  # assure the global_size is multiply of local_size
    # run the optimized kernel
    evt4 = prg.calcOrientationHist(queue, [global_size, 1, 1], [local_size, 1, 1], gpyr_buf, idx_buf, kpts_out3_buf, kpts_result2_buf,
                                   np.uint32(rows), np.uint32(cols), np.int32(firstOctave), np.uint32(ori_hist_bins),
                                   np.uint32(ori_radius), np.uint32(ori_sig_fctr), np.float32(ori_peak_ratio), np.uint32(kpts_result2.shape[0]))
    evt4.wait()
    kernel_time[3] += (1e-9 * (evt4.profile.end - evt4.profile.start))
    cl.enqueue_copy(queue, kpts_out3, kpts_out3_buf)
    cl.enqueue_copy(queue, idx, idx_buf)
    gpyr_buf.release()
    idx_buf.release()
    kpts_out3_buf.release()
    kpts_result2_buf.release()

    kpts_result3 = np.copy(kpts_out3[0:idx[0]])
    print('-- kernel calcOrientationHist finished')
    print('the number of final keypoints: {}'.format(kpts_result3.shape[0]))

    # pend to output list
    kpt_list.append(kpts_result3)
    print('')

print('')
namelist = ['DoGPyramid', 'findLocalExtrema', 'adjustLocalExtrema', 'calcOrientationHist']
for i in range(4):
    print("optimized kernel {} total running time: {}".format(namelist[i], kernel_time[i]))
    if i == 0:
        print('       cv2 build DoG Pyramid time: {}'.format(test_time))
print('')

# final keyoints
kpt_list = np.concatenate(kpt_list, axis=0)
print('total number of final keypoints: {}'.format(kpt_list.shape[0]))

# put results into cv2.keypoint
kpts = []
for i in range(kpt_list.shape[0]):
    kpts.append(cv2.KeyPoint(x=kpt_list[i][1], y=kpt_list[i][0], _size=kpt_list[i][3], _angle=kpt_list[i][5],
                             _response=kpt_list[i][4], _octave=kpt_list[i][2]))

# generate result image
print 'the final image with keypoints generated: PyOpenCL_result.jpg'
image = np.zeros_like(img)
# image = cv2.drawKeypoints(img, kpts, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image = cv2.drawKeypoints(img, kpts, image)
cv2.imwrite('PyOpenCL_result.jpg', image)

print('')
print 'running the OpenCV SIFT'
start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img, None)
end = time.time()
print("total time for OpenCV SIFT: {}".format(end - start))
image = cv2.drawKeypoints(img, kp, image)
cv2.imwrite('OpenCV_SIFT.jpg', image)
print 'the final image of OpenCV SIFT: OpenCV_SIFT.jpg'

print('')
print 'running the OpenCV SURF'
start = time.time()
surf = cv2.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img, None)
end = time.time()
print("total time for OpenCV SURF: {}".format(end - start))
image = cv2.drawKeypoints(img, kp, image)
cv2.imwrite('OpenCV_SURF.jpg', image)
print 'the final image of OpenCV SURF: OpenCV_SURF.jpg'
