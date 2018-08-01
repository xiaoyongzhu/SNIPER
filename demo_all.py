# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# SNIPER demo
# by Mahyar Najibi
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from configs.faster.default_configs import config, update_config, update_config_from_list
import mxnet as mx
import argparse
import sys
from train_utils.utils import create_logger, load_param
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pathlib2 as pathlib
from os.path import basename
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
from inference import Tester
from symbols.faster import *
from pytictoc import TicToc
from multiprocessing import Process
import pickle
import math
import cv2
import tqdm as tqdm
import numpy as np
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

t = TicToc()  # create instance of class

t.toc(restart=True)


def parser():
    arg_parser = argparse.ArgumentParser('SNIPER demo module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
                            default='configs/faster/sniper_res101_e2e_xview_fullimg_extractproposal.yml', type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--im_path', dest='im_path', help='Path to the image', type=str,
                            default='data/demo/9.tif')
    arg_parser.add_argument(
        '--output_folder', dest='output_folder', help='output folder', type=str)
    arg_parser.add_argument(
        "-o", "--output_file", default="predictions.txt", help="Filepath of desired output")
    return arg_parser.parse_args()


def draw_bboxes(img,boxes,classes,scores):
    """
    Draw bounding boxes on top of an image

    Args:
        img : Array of image to be modified
        boxes: An (N,4) array of boxes to draw, where N is the number of boxes.
        classes: An (N,1) array of classes corresponding to each bounding box.

    Outputs:
        An array of the same shape as 'img' with bounding boxes
            and classes drawn

    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2, h2 = (img.shape[0], img.shape[1])

    idx = 0

    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        c = classes[i]

        draw.text((xmin + 15, ymin + 15), str(int(c)))

        for j in range(4):
            if scores[i] > 0.5:
                draw.rectangle(((xmin + j, ymin + j), (xmax + j, ymax + j)), outline = "green")
            elif scores[i] > 0.3:
                draw.rectangle(((xmin + j, ymin + j), (xmax + j, ymax + j)), outline = "red")
            elif scores[i] > 0.1:
                draw.rectangle(((xmin + j, ymin + j), (xmax + j, ymax + j)), outline = "blue")
    return source


def chip_image(img, chip_size=(300, 300), img_name_folder=None):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    width, height, _ = img.shape
    wn, hn = chip_size
    width_chip_num = int(width / wn)
    height_chip_num = int(height/hn)
    print("padded image size is", width, height, "actual ratio is", width/wn,
          height/hn, "and round up to", width_chip_num, height_chip_num)
    images = np.zeros((int(width / wn) * int(height / hn), wn, hn, 3))
    k = 0
    for i in range(width_chip_num):
        for j in range(height_chip_num):

            chip = img[wn * i:wn * (i + 1), hn * j:hn * (j + 1), :3]
            image_folder = img_name_folder
            pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)
            filename = image_folder + "/" + str(i) + "_" + str(j) + ".jpg"
            arr2im = Image.fromarray(chip)
            arr2im.save(filename)
            images[k] = chip

            k = k + 1

    return width_chip_num, height_chip_num


# def roundup_to_num(x, target):
#     return int(math.ceil(x / float(target))) * int(target)


# def smart_chipping(origin_width, origin_height):
#     # 2300 will definitely work. Trying out 2400
#     tested_cpu_scoring_resolution = 1120
#     # smart chipping
#     stride = 16
#     max_cpu_scoring_resolution = roundup_to_num(
#         tested_cpu_scoring_resolution, stride)
#     max_resolution = max(origin_width, origin_height)
#     min_resolution = min(origin_width, origin_height)
#     if min_resolution > tested_cpu_scoring_resolution * 3 or max_resolution > tested_cpu_scoring_resolution * 5:
#         # TODO: resize the image to make sure the min resolution is within 3 * tested_cpu_scoring_resolution
#         pass

#     # TODO: re-write it using a more elegent
#     if max_resolution < max_cpu_scoring_resolution:
#         # only one chip
#         portion = roundup_to_num(max_resolution, stride)
#     elif max_resolution < max_cpu_scoring_resolution * 2:
#         # if possible, divide the image into 4 sub images
#         # the value needs to be divided by 64, since we will divide this number by two and the result needs to be divided by 32
#         portion = roundup_to_num(max_resolution, stride*2)/2
#     elif max_resolution < max_cpu_scoring_resolution * 3:
#         # if possible, divide the image into 4 sub images
#         # the value needs to be divided by 64, since we will divide this number by two and the result needs to be divided by 32
#         portion = roundup_to_num(max_resolution, stride*3)/3
#     elif max_resolution < max_cpu_scoring_resolution * 4:
#         portion = roundup_to_num(max_resolution, stride*4)/4
#     elif max_resolution < max_cpu_scoring_resolution * 5:
#         portion = roundup_to_num(max_resolution, stride*5)/5
#     else:
#         portion = max_cpu_scoring_resolution
#     return int(portion)


def generate_detections(width_chip_num, height_chip_num, detection_num, portion):
    ret_boxes = []
    ret_scores = []
    ret_classes = []
    conversion_dict = {0: "11", 1: "12", 2: "13", 3: "15", 4: "17", 5: "18", 6: "19", 7: "20", 8: "21", 9: "23", 10: "24", 11: "25", 12: "26", 13: "27", 14: "28", 15: "29", 16: "32", 17: "33", 18: "34", 19: "35", 20: "36", 21: "37", 22: "38", 23: "40", 24: "41", 25: "42", 26: "44", 27: "45", 28: "47", 29: "49",
                       30: "50", 31: "51", 32: "52", 33: "53", 34: "54", 35: "55", 36: "56", 37: "57", 38: "59", 39: "60", 40: "61", 41: "62", 42: "63", 43: "64", 44: "65", 45: "66", 46: "71", 47: "72", 48: "73", 49: "74", 50: "76", 51: "77", 52: "79", 53: "83", 54: "84", 55: "86", 56: "89", 57: "91", 58: "93", 59: "94"}

    for width_chip_num_index in range(width_chip_num):
        for height_chip_num_index in range(height_chip_num):

            dets_nms = chipped_image_scoring(
                width_chip_num_index, height_chip_num_index, portion)

            # dets_nms format:
            # a length 60 array, with each element being a dict representing each class
            # the coordinates are in (xmin, ymin, xmax, ymax, confidence) format. The coordinates are not normalized
            # one sample is: [290.09448    439.60617    333.31235    461.8115       0.94750994]
            # below iterates class by class
            image_detection_num = 0

            for index_class in range(len(dets_nms)):
                # for each class
                single_class_nms = dets_nms[index_class][0]
                # print("single_class_nms", single_class_nms)
                image_detection_num += len(single_class_nms)
                if len(single_class_nms) != 0:
                    # print("foudn class", index_class,
                    #       "number of objects", len(single_class_nms))
                    # print(single_class_nms)
                    for index_single_class_nms in range(min(len(single_class_nms), detection_num)):
                        # print("index_class,index_single_class_nms", index_class,index_single_class_nms, )
                        # get all the element other than the last one
                        ret_boxes.append(
                            single_class_nms[index_single_class_nms][:-1])
                        ret_scores.append(
                            single_class_nms[index_single_class_nms][-1])  # last element
                        # we predict 61 classes... So fix it here using some hacky way
                        ret_classes.append(conversion_dict[index_class-1])
            # pad zeros
            # print("1st: len(ret_boxes), image_detection_num", len(ret_boxes), image_detection_num)
            if image_detection_num <= detection_num:
                for index_element in range(int(detection_num - image_detection_num)):
                    # get all the element other than the last one
                    ret_boxes.append(np.zeros((4,), dtype=np.float32))
                    ret_scores.append(0)  # last element
                    ret_classes.append(0)
            else:
                print("~~~~~ too many predictions ~~~~~~~~~~~~~~~")
            print("len(ret_boxes), image_detection_num",
                  len(ret_boxes), image_detection_num)

            print('testing image width {} height {}, detection number {}'.format(
                width_chip_num_index, height_chip_num_index, image_detection_num))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # visualize
            # don't show the final images.
            # show_boxes(im, dets_nms, classes, 1, show_image = False, img_save_name = str(idx) + ".png")

    ret_boxes = np.squeeze(np.array(ret_boxes))
    ret_scores = np.squeeze(np.array(ret_scores))
    ret_classes = np.squeeze(np.array(ret_classes))

    return ret_boxes, ret_scores, ret_classes




def chipped_image_scoring(width_chip_num_index, height_chip_num_index, portion):

    args = parser()
    image_folder = args.output_folder
    im_path = os.path.join(image_folder, str(width_chip_num_index) + "_" +
                           str(height_chip_num_index) + ".jpg")

    for scale_index in range(len(config.TEST.SCALES)):
        # print("scale_index", scale_index)
        cmd = "python demo.py --use_gpu True --im_path " + im_path + \
            " --cfg configs/faster/sniper_res101_e2e_xview_fullimg_extractproposal.yml --scale_index " + \
            str(scale_index) + " --chip_size " + str(portion)
        os.system(cmd)
    # Sequentially do detection over scales
    # NOTE: if you want to perform detection on multiple images consider using main_test which is parallel and faster
    all_detections = []

    for scale_index in range(len(config.TEST.SCALES)):
        file_name = os.path.join(image_folder, str(scale_index)+".pkl")
        with open(file_name, 'rb') as handle:
            res = pickle.load(handle)
        all_detections.append(res)

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Get image dimensions
    width, height = Image.open(im_path).size

    # Pack image info
    roidb = [{'image': im_path, 'width': width,
              'height': height, 'flipped': False}]

    # Pack db info
    db_info = EasyDict()
    db_info.name = 'coco'
    db_info.result_path = 'data/demo'

    # Categories the detector trained for:
    db_info.classes = ['Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter', 'Passenger Vehicle', 'Small Car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box', 'Truck Tractor', 'Trailer', 'Truck w/Flatbed', 'Truck w/Liquid', 'Crane Truck', 'Railway Vehicle', 'Passenger Car', 'Cargo Car', 'Flat Car', 'Tank car', 'Locomotive', 'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge', 'Fishing Vessel', 'Ferry', 'Yacht','Container Ship', 'Oil Tanker', 'Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker', 'Straddle Carrier', 'Mobile Crane', 'Dump Truck', 'Haul Truck', 'Scraper/Tractor', 'Front loader/Bulldozer', 'Excavator', 'Cement Mixer', 'Ground Grader', 'Hut/Tent', 'Shed', 'Building', 'Aircraft Hangar', 'Damaged Building', 'Facility', 'Construction Site', 'Vehicle Lot', 'Helipad', 'Storage Tank', 'Shipping container lot', 'Shipping Container', 'Pylon', 'Tower']
    db_info.num_classes = len(db_info.classes)

    # Aggregate results from multiple scales and perform NMS
    # Create the tester
    tester = Tester(None, db_info, roidb, None, cfg=config, batch_size=1)
    file_name, out_extension = os.path.splitext(os.path.basename(im_path))
    all_detections = tester.aggregate(all_detections, vis=True, cache_name=None, vis_path='./data/demo/',
                                      vis_name='{}_detections'.format(file_name), vis_ext=out_extension)
    # all_detections = tester.aggregate(all_detections, vis=True, cache_name=None, vis_path='./data/demo/',
    #                                      vis_name='{}_detections_1'.format(file_name), vis_ext='jpg')

    return all_detections

def main():
    args = parser()
    update_config(args.cfg)

    im = cv2.cvtColor(cv2.imread(args.im_path), cv2.COLOR_BGR2RGB)

    arr = np.array(im)
    origin_width, origin_height, _ = arr.shape
    # portion = smart_chipping(origin_width, origin_height)
    # portion = 1120
    portion = 1080

    cwn, chn = (portion, portion)
    wn, hn = (int(origin_width / cwn), int(origin_height / chn))
    padding_y = int(math.ceil(float(origin_height)/chn) * chn - origin_height)
    padding_x = int(math.ceil(float(origin_width)/cwn) * cwn - origin_width)
    print("padding_y,padding_x, origin_height, origin_width",
          padding_y, padding_x, origin_height, origin_width)
    # top, bottom, left, right - border width in number of pixels in corresponding directions
    im = cv2.copyMakeBorder(im, 0, padding_x, 0, padding_y,
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # the section below could be optimized. but basically the idea is to re-calculate all the values
    arr = np.array(im)
    width, height, _ = arr.shape
    cwn, chn = (portion, portion)
    wn, hn = (int(width / cwn), int(height / chn))
    img_name_folder = args.output_folder
    width_chip_num, height_chip_num = chip_image(
        im, (portion, portion), img_name_folder)

    num_preds = int(6000 * math.ceil(float(portion)/400))
    boxes, scores, classes = generate_detections(
        width_chip_num, height_chip_num, num_preds, portion)
    print("boxes shape is", boxes.shape, "wn, hn",
          wn, hn, "width, height", width, height)
    bfull = boxes.reshape((wn, hn, num_preds, 4))

    for i in range(wn):
        for j in range(hn):
            bfull[i, j, :, 0] += j*cwn
            bfull[i, j, :, 2] += j*cwn

            bfull[i, j, :, 1] += i*chn
            bfull[i, j, :, 3] += i*chn

            # clip values
            bfull[i, j, :, 0] = np.clip(bfull[i, j, :, 0], 0, origin_height)
            bfull[i, j, :, 2] = np.clip(bfull[i, j, :, 2], 0, origin_height)

            bfull[i, j, :, 1] = np.clip(bfull[i, j, :, 1], 0, origin_width)
            bfull[i, j, :, 3] = np.clip(bfull[i, j, :, 3], 0, origin_width)

    bfull = bfull.reshape((hn * wn, num_preds, 4))
    scores = scores.reshape((hn * wn, num_preds))
    classes = classes.reshape((hn * wn, num_preds))

    # only display boxes with confidence > .5
    bs = bfull[scores > 0.1]
    cs = classes[scores > 0.1]
    ss = scores[scores > 0.1]
    output_im_path = os.path.splitext(os.path.basename(args.im_path))[0]+".jpg"
    draw_bboxes(arr, bs, cs,ss).save(os.path.join(args.output_folder, output_im_path))

    score_thres = 0.1
    # if bs.shape[0] > scoring_line_threshold:
    # too many predictions, we should trim the low confidence ones
    with open(args.output_file, 'w') as f:
        for i in range(bfull.shape[0]):
            for j in range(bfull[i].shape[0]):
                # box should be xmin ymin xmax ymax
                box = bfull[i, j]
                class_prediction = classes[i, j]
                score_prediction = scores[i, j]
                if int(class_prediction) != 0 and score_prediction >= score_thres:
                    f.write('%d %d %d %d %d %f \n' %
                            (box[0], box[1], box[2], box[3], int(class_prediction), score_prediction))

    print('done')


if __name__ == '__main__':
    main()

