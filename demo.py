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
from train_utils.utils import create_logger, load_param
import os
from PIL import Image
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
from inference import Tester
from symbols.faster import resnet_mx_50_e2e, resnet_mx_101_e2e
import pickle
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def parser():
    arg_parser = argparse.ArgumentParser('SNIPER demo module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
                            default='configs/faster/sniper_res101_e2e_xview_fullimg_extractproposal.yml', type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--im_path', dest='im_path', help='Path to the image', type=str,
                            default='data/demo/demo.jpg')
    arg_parser.add_argument('--use_gpu', dest='use_gpu', help='use GPU for inference', type=bool,
                            default=False)
    arg_parser.add_argument(
        '--scale_index', dest='scale_index', help='scale index', type=int)
    arg_parser.add_argument(
        '--chip_size', dest='chip_size', help='chip_size', type=int)
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()


def main():
    args = parser()
    update_config(args.cfg)

    # Use just the first GPU for demo
    if args.use_gpu:
        context = [mx.gpu(int(config.gpus[0]))]
    else:
        context = [mx.cpu()]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Get image dimensions
    width, height = Image.open(args.im_path).size

    # Pack image info
    roidb = [{'image': args.im_path, 'width': width,
              'height': height, 'flipped': False}]

    # Creating the Logger
    logger, output_path = create_logger(
        config.output_path, args.cfg, config.dataset.image_set)

    # Pack db info
    db_info = EasyDict()
    db_info.name = 'coco'
    db_info.result_path = 'data/demo'

    # Categories the detector trained for:
    db_info.classes = ['Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter', 'Passenger Vehicle', 'Small Car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box', 'Truck Tractor', 'Trailer', 'Truck w/Flatbed', 'Truck w/Liquid', 'Crane Truck', 'Railway Vehicle', 'Passenger Car', 'Cargo Car', 'Flat Car', 'Tank car', 'Locomotive', 'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge', 'Fishing Vessel', 'Ferry', 'Yacht',
                       'Container Ship', 'Oil Tanker', 'Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker', 'Straddle Carrier', 'Mobile Crane', 'Dump Truck', 'Haul Truck', 'Scraper/Tractor', 'Front loader/Bulldozer', 'Excavator', 'Cement Mixer', 'Ground Grader', 'Hut/Tent', 'Shed', 'Building', 'Aircraft Hangar', 'Damaged Building', 'Facility', 'Construction Site', 'Vehicle Lot', 'Helipad', 'Storage Tank', 'Shipping container lot', 'Shipping Container', 'Pylon', 'Tower']
    db_info.num_classes = len(db_info.classes)

    # Create the model
    sym_def = eval('{}.{}'.format(config.symbol, config.symbol))
    sym_inst = sym_def(n_proposals=400, test_nbatch=1)
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)
    test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=1, nGPUs=1, threads=1,
                               crop_size=None, test_scale=config.TEST.SCALES[args.scale_index],
                               num_classes=db_info.num_classes)
    # Create the module
    shape_dict = dict(test_iter.provide_data_single)
    sym_inst.infer_shape(shape_dict)
    mod = mx.mod.Module(symbol=sym,
                        context=context,
                        data_names=[k[0]
                                    for k in test_iter.provide_data_single],
                        label_names=None)
    mod.bind(test_iter.provide_data,
             test_iter.provide_label, for_training=False)

    # Initialize the weights
    model_prefix = os.path.join(output_path, args.save_prefix)
    arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                        convert=True, process=True)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # Create the tester
    tester = Tester(mod, db_info, roidb, test_iter, cfg=config, batch_size=1)

    # Set tester scale
    # print("args.chip_size * config.TEST.SCALES[args.scale_index]",args.chip_size * config.TEST.SCALES[args.scale_index], args.chip_size ,config.TEST.SCALES[args.scale_index])
    tester.set_scale(config.TEST.SCALES[args.scale_index])
    # Perform detection

    res = tester.get_detections(vis=False, evaluate=False, cache_name=None)
    folder_name = os.path.dirname(args.im_path)
    file_name = os.path.join(folder_name, str(args.scale_index)+".pkl")
    with open(file_name, 'wb') as handle:
        pickle.dump(res, handle)


if __name__ == '__main__':
    main()
