import sys
import os
import glob
import argparse as ap
import numpy as np

if __name__ == "__main__":
    # Parse the command line arguments
    #test_im_path = "../data/dataset/CarData/TestImages/test-16.pgm"
    parser = ap.ArgumentParser()
    #parser.add_argument('-i', "--image", help="Path to the test image", default='../data/dataset/PNGImages/TestImages/test-16.png', required=True)
    parser.add_argument('-i', '--imgpath', help='Path to the test image',
                        default='../data/dataset/PNGImages/TestImages', required=False)
    parser.add_argument('-g', '--gtboxpath', help='Path to the ground-truth bounding box',
                        default='../data/dataset/Annotations/TestImages', required=False)
    parser.add_argument('-c', '--gtconvert', help='Path to save the converted ground-truth txt files',
                        default='../data/TestImages/ground-truth', required=False)
    args = vars(parser.parse_args())

    test_im_path = args["imgpath"]
    gtbox_path = args["gtboxpath"]
    gt_conv_path = args["gtconvert"]

    for im_path in glob.glob(os.path.join(test_im_path, "*")):

        im_name_tmp = im_path.split('\\')
        im_name = im_name_tmp[-1][0:-4] + '.txt'
        txt_path = os.path.join(gtbox_path, im_name)


        with open(txt_path, 'r') as gt_file:

            gt_pos = []

            for line in gt_file:
                data = line.strip()
                #data = line.strip('').split()
                ld = len(data)

                xmin_id = data.find("(Xmin, Ymin)")

                if xmin_id > 0:
                    pos_data = data[xmin_id+30:]
                    jian_id = pos_data.find(") - ")

                    pos01_data = pos_data[1:jian_id]
                    pos02_data = pos_data[jian_id + 5:-1]

                    dou01_id = pos01_data.find(",")
                    dou02_id = pos02_data.find(",")

                    left = pos01_data[0:dou01_id]
                    top = pos01_data[dou01_id+2:]
                    right = pos02_data[0:dou02_id]
                    bottom = pos02_data[dou02_id+2:]

                    gt_pos.append(("carSide", left, top, right, bottom))

            gt_conv_txt_path = os.path.join(gt_conv_path, im_name)
            np.savetxt(gt_conv_txt_path, gt_pos, fmt='%s')


