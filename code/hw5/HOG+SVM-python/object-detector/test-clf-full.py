from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage import feature
import joblib
import cv2
import argparse as ap
import glob
import os
from nms import nms
from config import *
import numpy as np


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


if __name__ == "__main__":
    # Parse the command line arguments
    #test_im_path = "../data/dataset/CarData/TestImages/test-16.pgm"
    parser = ap.ArgumentParser()
    #parser.add_argument('-i', "--image", help="Path to the test image", default='../data/dataset/PNGImages/TestImages/test-16.png', required=True)
    parser.add_argument('-i', '--imgpath', help='Path to the test image',
                        default='../data/dataset/PNGImages/TestImages', required=False)
    parser.add_argument('-s', '--detboxpath', help='Path to the save output bounding box',
                        default='../data/TestImages/detection-results', required=False)
    parser.add_argument('-d', '--downscale', help="Downscale ratio", default=1.25, type=int)
    parser.add_argument('-v', '--visualize', help="Visualize the sliding window", action="store_true")
    args = vars(parser.parse_args())

    test_im_path = args["imgpath"]
    detbox_path = args["detboxpath"]

    min_wdw_sz = (100, 40)
    step_size = (10, 10)
    downscale = args['downscale']
    visualize_det = args['visualize']
    #visualize_det = True

    print(model_path)

    # Load the classifier
    clf = joblib.load(model_path)
    #model_coef = clf.coef_
    #np.savetxt('model_coef.txt', model_coef, fmt='%f')

    # Read the image
    for im_path in glob.glob(os.path.join(test_im_path, "*")):
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

        print(im_path)

        # List to store the detections
        detections = []
        # The current scale of the image
        scale = 0
        # Downscale the image and iterate

        for im_scaled in pyramid_gaussian(im, downscale=downscale):
            # This list contains detections at the current scale
            cd = []
            # If the width or height of the scaled image is less than
            # the width or height of the window, then end the iterations.
            srow = im_scaled.shape[0]
            scol = im_scaled.shape[1]
            if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue
                # Calculate the HOG features
                #fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize)
                fd = feature.hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False)
                fd_reshape = fd.reshape(1, -1)
                #pred = clf.predict(numpy.array(fd).reshape(1, -1))
                pred = clf.predict(fd_reshape)

                if pred == 1:
                    print("Detection:: Location -> ({}, {})".format(x, y))
                    print("Scale ->  {} | Confidence Score {} \n".format(scale, clf.decision_function(fd_reshape)))
                    # detections.append((x, y, clf.decision_function(fd_reshape),
                    #    int(min_wdw_sz[0]*(downscale**scale)),
                    #    int(min_wdw_sz[1]*(downscale**scale))))
                    detections.append((int(x*(downscale**scale)), int(y*(downscale**scale)), clf.decision_function(fd_reshape),
                        int(min_wdw_sz[0]*(downscale**scale)),
                        int(min_wdw_sz[1]*(downscale**scale))))
                '''
                    cd.append(detections[-1])
                # If visualize is set to true, display the working
                # of the sliding window
                if visualize_det:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _  in cd:
                        # Draw the detections at this scale
                        x1_02 = int(x1/(downscale**scale))
                        y1_02 = int(y1/(downscale**scale))
                        cv2.rectangle(clone, (x1_02, y1_02), (x1_02 + im_window.shape[1], y1_02 +
                            im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                        im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Sliding Window in Progress", clone)
                    cv2.waitKey(30)
                    '''
            # Move the the next scale
            scale+=1
        '''
        # Display the results before performing NMS
        clone = im.copy()
        for (x_tl, y_tl, _, w, h) in detections:
            # Draw the detections
            cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
        cv2.imshow("Raw Detections before NMS", im)
        cv2.waitKey()
        '''

        # Perform Non Maxima Suppression
        detections = nms(detections, threshold)

        print("after NMS, the detections=")
        print(detections)

        if detections != None:
            #continue
            detections_output = []
            im_name_tmp = im_path.split('\\')
            im_name = im_name_tmp[-1][0:-4]+'.txt'
            txt_path = os.path.join(detbox_path, im_name)

            # Display the results after performing NMS
            for (x_tl, y_tl, boxp, w, h) in detections:
                boxp_out = np.round(boxp[0], 6)
                detections_output.append(('carSide', boxp_out, x_tl, y_tl, x_tl+w, y_tl+h))
                np.savetxt(txt_path, detections_output, fmt='%s')

        # Save the detections
                print("detected bounding boxs are saved in {}".format(txt_path))

    print("Completed detection of bounding boxs from test images")

