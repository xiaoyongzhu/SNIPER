# ---------------------------------------------------------------
# SNIPER: Efficient Multi-scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified from https://github.com/msracver/Deformable-ConvNets
# Modified by Mahyar Najibi
# ---------------------------------------------------------------
import cPickle
import os
import json
import numpy as np

from imdb import IMDB
# coco api
from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval
from mask.mask_voc2coco import mask_voc2coco
from bbox.bbox_transform import clip_boxes, bbox_overlaps_py
import multiprocessing as mp


def coco_results_one_category_kernel(data_pack):
    cat_id = data_pack['cat_id']
    ann_type = data_pack['ann_type']
    binary_thresh = data_pack['binary_thresh']
    all_im_info = data_pack['all_im_info']
    boxes = data_pack['boxes']
    if ann_type == 'bbox':
        masks = []
    elif ann_type == 'segm':
        masks = data_pack['masks']
    else:
        print 'unimplemented ann_type: ' + ann_type
    cat_results = []
    for im_ind, im_info in enumerate(all_im_info):
        index = im_info['index']
        dets = boxes[im_ind].astype(np.float)
        if len(dets) == 0:
            continue
        scores = dets[:, -1]
        if ann_type == 'bbox':
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'bbox': [round(xs[k], 1), round(ys[k], 1), round(ws[k], 1), round(hs[k], 1)],
                       'score': round(scores[k], 8)} for k in xrange(dets.shape[0])]
        elif ann_type == 'segm':
            width = im_info['width']
            height = im_info['height']
            dets[:, :4] = clip_boxes(dets[:, :4], [height, width])
            mask_encode = mask_voc2coco(masks[im_ind], dets[:, :4], height, width, binary_thresh)
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'segmentation': mask_encode[k],
                       'score': scores[k]} for k in xrange(len(mask_encode))]
        cat_results.extend(result)
    return cat_results


class coco(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None, mask_size=-1, binary_thresh=None, load_mask=False):
        """
        fill basic information to initialize imdb
        :param image_set: train2014, val2014, test2015
        :param root_path: 'data', will write 'rpn_data', 'cache'
        :param data_path: 'data/coco'
        """
        super(coco, self).__init__('COCO', image_set, root_path, data_path, result_path)
        self.root_path = root_path
        self.data_path = data_path
        self.coco = COCO(self._get_ann_file())

        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh
        self.load_mask = load_mask

        # deal with data name
        view_map = {'minival2014': 'val2014',
                    'sminival2014': 'val2014',
                    'valminusminival2014': 'val2014',
                    'test-dev2015': 'test2015'}

        self.data_name = view_map[image_set] if image_set in view_map else image_set

    def _get_ann_file(self):
        """ self.data_path / annotations / instances_train2014.json """
        prefix = 'instances' if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.data_path, 
                            prefix + '_' + self.image_set + '.json')

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        # too_many_gt_img = set([22530, 21163, 22532, 55301, 51119, 63502, 21165, 63505, 22531, 43028, 27310, 63513, 43035, 63519, 53936, 30310, 63524, 48817, 30006, 55020, 55211, 27316, 25507, 57013, 65600, 65601, 55307, 65604, 65605, 65606, 65609, 65610, 65611, 65612, 65614, 65615, 65616, 65617, 25519, 65619, 65620, 65621, 65622, 10327, 10328, 3208, 65627, 65629, 65630, 65631, 65632, 65633, 65634, 65635, 65636, 65637, 65638, 65639, 23228, 55402, 55403, 8300, 59501, 59502, 51311, 55408, 6504, 59506, 59507, 59508, 22206, 59511, 20600, 67604, 20602, 20603, 59516, 59517, 59518, 17429, 20608, 59521, 59524, 59525, 19600, 20615, 38040, 8329, 20618, 44055, 20620, 59535, 45200, 45201, 6510, 20631, 2200, 45209, 49306, 45212, 45213, 45214, 2207, 2208, 45217, 45218, 49315, 49320, 2217, 18119, 20737, 45229, 30917, 2223, 2224, 14515, 30901, 30012, 2232, 2233, 30908, 35005, 2238, 30911, 2240, 30913, 30914, 22219, 30835, 35013, 30918, 2247, 30922, 30923, 4300, 35021, 35022, 4303, 35024, 4305, 22034, 4307, 4308, 59605, 4311, 55030, 4315, 4316, 4317, 30841, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 22225, 4328, 4329, 4330, 4331, 4332, 4333, 20718, 37103, 22035, 24729, 4338, 22227, 20724, 37109, 37110, 37111, 20730, 14719, 37116, 37117, 22229, 37121, 53506, 49412, 61701, 10503, 10504, 10505, 49418, 18700, 10509, 10510, 10511, 18706, 10515, 10516, 38617, 10521, 10522, 25534, 3802, 31006, 10527, 10528, 39201, 39202, 55003, 39204, 39205, 10534, 10535, 39208, 55004, 39210, 39211, 10540, 39213, 39215, 51504, 59613, 39219, 8500, 43318, 39224, 39225, 39226, 39227, 39228, 39229, 39230, 39231, 39232, 8513, 39234, 39235, 25524, 4833, 51530, 51531, 51532, 51533, 51537, 51538, 55011, 24919, 41304, 3812, 41306, 41307, 24924, 55013, 41313, 41314, 41316, 6501, 6502, 57018, 2408, 41321, 6507, 6508, 6509, 14702, 2415, 41329, 14707, 6516, 35902, 6521, 6522, 61819, 14716, 14717, 6526, 6527, 6528, 14723, 6532, 6533, 22040, 26200, 14729, 59532, 25533, 30625, 30022, 23619, 31127, 57025, 51613, 55024, 22041, 16805, 12710, 55025, 12715, 12716, 12718, 12719, 12720, 12721, 55731, 23625, 37306, 37307, 30336, 37313, 45506, 37622, 32000, 59618, 55031, 45516, 45517, 45520, 45521, 37331, 37332, 45526, 55033, 55432, 45532, 27101, 61706, 22221, 22608, 27108, 59619, 6908, 35308, 35309, 27118, 55037, 27120, 27121, 27122, 23635, 35316, 49918, 33637, 35323, 47612, 47616, 32512, 30331, 35331, 47620, 16904, 21001, 21002, 21003, 21004, 23613, 21006, 21007, 21008, 21009, 21010, 21012, 27104, 21014, 21015, 21017, 21019, 21020, 21021, 21022, 21023, 21024, 45602, 21027, 21028, 21029, 45606, 45607, 16936, 21033, 21034, 17614, 6703, 30330, 17613, 26718, 37641, 66104, 19003, 19004, 66110, 19009, 25154, 38618, 30912, 26722, 35408, 49325, 47203, 35412, 35413, 51712, 35417, 51124, 47205, 611, 22205, 30904, 25509, 33400, 32532, 33406, 33407, 33410, 25529, 17029, 11410, 17031, 17032, 17034, 33421, 33422, 41617, 21138, 37523, 25527, 41623, 41624, 33433, 59625, 41629, 53918, 41631, 59225, 59504, 53924, 57019, 41638, 55409, 19114, 41643, 21164, 13426, 9729, 39709, 19120, 41650, 41651, 41652, 27317, 19126, 59626, 28415, 19132, 19133, 4802, 4803, 4804, 4805, 4806, 4809, 4810, 4811, 4812, 4816, 4817, 30915, 4819, 55611, 4822, 4823, 4824, 4825, 4826, 4827, 4829, 4830, 4831, 4832, 33505, 4834, 59515, 4836, 4839, 4840, 65400, 37621, 45814, 39721, 59512, 37625, 37626, 37627, 37628, 15102, 15103, 37632, 59520, 37634, 15107, 15108, 37334, 1153, 37640, 15113, 2829, 15119, 31504, 27409, 30311, 27414, 39703, 43801, 27418, 43804, 2845, 30919, 27425, 39715, 2852, 20614, 37340, 39722, 3207, 39727, 39728, 11400, 21301, 65509, 18229, 33600, 33601, 21314, 20619, 21032, 41800, 11404, 41802, 11405, 55039, 40419, 30840, 11406, 7000, 41822, 7007, 7008, 33633, 33634, 33635, 33636, 41829, 33638, 33639, 19601, 33641, 58218, 33643, 33644, 62610, 7022, 7023, 64832, 58225, 58226, 11411, 58228, 7029, 60905, 39800, 58233, 58234, 58235, 60907, 19606, 19602, 20625, 48011, 30530, 59513, 45208, 48019, 5012, 5013, 5015, 5016, 5017, 55610, 5020, 5021, 5022, 25503, 25504, 21035, 5026, 5027, 5028, 5029, 25510, 25511, 25512, 24732, 25514, 25515, 25517, 25518, 11421, 25520, 3912, 27123, 30924, 6302, 41801, 25528, 3001, 25530, 25532, 46922, 3006, 3007, 3008, 32432, 3013, 7110, 2209, 3017, 51703, 3021, 3024, 3027, 3028, 3030, 3033, 3034, 59215, 26106, 35806, 51707, 17423, 35324, 6312, 21331, 2216, 44033, 44034, 32427, 44039, 44040, 44041, 44045, 44046, 44047, 44048, 17426, 17427, 44052, 44053, 44054, 9239, 17432, 17433, 17434, 17435, 16811, 17441, 59227, 37925, 30329, 17447, 9256, 22023, 3116, 7214, 51114, 37936, 21000, 23602, 23603, 23604, 23605, 23608, 23609, 35900, 35901, 23614, 23615, 41824, 22027, 23620, 23621, 35913, 23626, 6515, 23628, 15437, 23630, 23631, 19640, 23634, 15203, 13400, 23641, 22031, 1118, 1119, 48224, 22032, 1124, 1125, 48230, 22033, 13421, 25710, 25711, 1138, 13427, 58217, 13432, 1145, 1146, 25508, 17428, 22037, 1152, 3201, 3202, 20331, 22038, 1159, 1160, 11401, 11402, 11403, 3212, 3213, 3214, 11407, 11408, 11409, 3218, 3219, 11412, 11413, 38038, 38039, 11416, 11417, 11418, 11419, 19612, 3229, 11422, 1221, 3234, 3235, 11428, 19621, 19622, 51118, 17607, 19628, 19630, 19631, 17608, 19634, 19315, 19638, 17609, 1208, 60601, 1211, 58229, 13504, 1217, 13509, 13511, 1224, 13513, 1227, 17612, 1229, 1230, 4301, 58232, 17618, 17619, 17620, 4302, 30314, 17624, 58212, 24728, 52431, 17628, 46619, 4304, 17634, 17635, 42219, 42225, 21026, 38136, 59604, 26213, 38139, 55008, 38142, 38143, 38144, 38145, 38146, 38147, 32006, 65409, 55009, 9815, 30325, 25902, 25903, 30000, 25905, 25907, 25908, 9728, 25911, 64824, 25914, 25915, 25916, 25917, 65401, 25920, 25921, 25923, 46909, 25926, 25927, 25928, 25929, 25932, 50509, 50510, 38223, 50514, 50515, 38228, 50519, 50520, 50521, 38234, 38235, 50525, 50526, 50527, 25147, 23909, 30315, 15933, 24733, 59624, 40310, 12521, 51125, 26711, 12522, 48515, 55205, 64027, 65431, 28226, 27311, 59514, 56900, 4335, 32438, 56901, 59632, 46500, 46501, 4337, 25500, 38314, 58219, 25501, 31007, 25502, 7608, 30909, 46525, 46526, 46527, 46528, 46529, 46532, 46533, 46534, 25505, 54728, 25506, 51128, 40405, 55204, 24026, 24029, 51109, 37115, 9703, 60904, 53500, 60906, 9707, 9708, 9709, 9715, 13812, 13813, 13814, 25513, 9721, 9722, 26107, 13820, 13821, 13822, 9727, 13824, 13825, 13826, 13827, 22020, 22021, 22022, 9735, 46600, 46601, 46602, 46603, 22028, 22029, 46606, 46607, 46608, 46609, 46610, 46611, 28421, 47022, 52425, 46615, 46616, 46617, 47023, 14714, 47024, 58223, 46635, 47026, 46638, 46639, 46640, 46641, 47027, 40502, 49417, 47028, 28220, 28221, 28222, 16309, 28225, 67507, 1604, 51126, 45608, 9800, 1610, 56907, 56908, 40529, 40530, 56915, 61710, 40535, 9816, 26201, 50516, 49423, 22215, 26206, 26207, 61703, 26212, 34405, 38502, 38503, 56936, 56937, 26719, 26219, 38508, 38509, 56942, 56943, 56944, 38513, 38514, 38515, 38517, 30326, 30328, 38521, 38522, 25535, 30332, 38525, 61704, 30335, 38528, 7809, 38531, 30340, 38534, 30343, 38537, 38538, 38540, 38541, 8512, 56249, 31001, 61705, 61722, 3807, 30308, 26224, 3014, 30321, 48808, 61716, 57002, 3015, 48813, 57006, 48815, 48816, 44721, 48818, 14005, 9809, 22200, 22201, 22202, 14011, 22204, 22203, 9918, 22207, 22208, 22209, 22210, 22211, 22212, 22213, 22214, 10529, 22216, 22218, 5835, 18125, 22222, 22223, 22224, 5841, 22226, 39203, 38612, 38613, 22230, 22231, 22232, 55001, 55002, 3803, 3804, 28408, 59505, 55007, 3808, 3809, 3811, 16100, 3813, 55014, 55015, 16106, 55019, 42732, 55021, 16112, 3825, 10541, 55029, 16118, 16119, 16120, 16121, 55034, 16124, 16125, 16126, 16127, 55040, 26724, 16130, 16131, 16132, 41318, 33632, 30337, 39212, 32001, 28428, 28430, 28431, 59519, 30339, 55435, 48918, 14105, 30612, 56914, 22039, 26725, 14112, 30342, 30503, 30504, 19633, 33415, 30510, 33416, 18228, 46901, 46902, 46904, 46907, 46908, 41610, 46910, 59200, 46913, 46914, 20725, 46916, 55606, 59206, 55112, 14724, 59213, 59214, 48822, 59219, 26113, 46933, 46934, 55609, 3929, 59226, 46939, 46940, 8506, 59230, 59231, 59232, 59235, 44731, 44904, 44905, 44906, 44907, 20332, 20333, 55613, 33427, 64830, 36727, 64031, 55161, 65403, 27127, 55168, 13505, 25910, 36742, 65417, 49036, 44941, 53912, 38804, 30316, 21145, 11415, 30619, 39209, 51104, 38817, 38818, 51108, 47013, 47014, 51113, 47018, 47019, 16300, 16301, 16302, 16303, 16304, 16306, 16307, 16308, 47029, 16310, 51127, 16312, 67513, 51131, 51132, 67517, 30916, 61700, 12235, 25933, 12241, 59526, 25934, 13433, 53212, 19639, 32421, 53216, 53217, 22026, 53219, 53220, 53221, 53222, 65511, 53226, 25525, 53229, 53230, 53231, 53233, 13434, 20731, 34808, 34809, 59616, 32426])

        # if index in too_many_gt_img:
        # print(image_ids)
        # image_ids = list(set(image_ids) - set(too_many_gt_img))
        return image_ids

    def image_path_from_index(self, index):
        """ example: images / train2014 / COCO_train2014_000000119993.jpg """
        # filename = 'COCO_%s_%012d.jpg' % (self.data_name, index)
        filename = index
        # image_path = os.path.join(self.data_path, 'images', self.data_name, filename)
        image_path = os.path.join(self.data_path, self.data_name, filename)

        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        index_file = os.path.join(self.cache_path, self.name + '_index_roidb.pkl')
        sindex_file = os.path.join(self.cache_path, self.name + '_sindex_roidb.pkl')
        if os.path.exists(cache_file) and os.path.exists(index_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            with open(index_file, 'rb') as fid:
                self.image_set_index = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = []
        valid_id = []
        vids = []
        ct = 0
        
        for index in self.image_set_index:
            roientry,flag = self._load_coco_annotation(index)
            if flag:
                gt_roidb.append(roientry)
                valid_id.append(index)
                vids.append(ct)
            ct = ct + 1
        self.image_set_index = valid_id

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        with open(index_file, 'wb') as fid:
            cPickle.dump(valid_id, fid, cPickle.HIGHEST_PROTOCOL)
        with open(sindex_file, 'wb') as fid:
            cPickle.dump(vids, fid, cPickle.HIGHEST_PROTOCOL)

        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_coco_annotation(self, index):
        def _polys2boxes(polys):
            boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
            for i in range(len(polys)):
                poly = polys[i]
                x0 = min(min(p[::2]) for p in poly)
                x1 = max(max(p[::2]) for p in poly)
                y0 = min(min(p[1::2]) for p in poly)
                y1 = max(max(p[1::2]) for p in poly)
                boxes_from_polys[i, :] = [x0, y0, x1, y1]
            return boxes_from_polys
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: roidb entry
        """
        
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=True)
        objsc = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        valid_objsc = []
        for obj in objsc:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objsc.append(obj)

        objs = valid_objs
        objc = valid_objsc
        num_objs = len(objs)
        num_objsc = len(objsc)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        boxesc = np.zeros((num_objsc, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        #for ix, obj in enumerate(objsc):
        #    boxesc[ix, :] = obj['clean_bbox']

        for ix, obj in enumerate(objs):
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            if obj['iscrowd']:
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]

        flag = True

        roi_rec = {'image': self.image_path_from_index(im_ann['file_name']),
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'boxesc': boxesc,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'flipped': False}
        if self.load_mask:
            # we only care about valid polygons

            segs = []
            for obj in objs:
                if not isinstance(obj['segmentation'], list):
                    # This is a crowd box
                    segs.append([])
                else:
                    segs.append([np.array(p) for p in obj['segmentation'] if len(p)>=6])
            
            roi_rec['gt_masks'] =  segs

            # Uncomment if you need to compute gts based on segmentation masks
            # seg_boxes = _polys2boxes(segs)
            # roi_rec['mask_boxes'] = seg_boxes
        return roi_rec, flag

    

    def evaluate_detections(self, detections, ann_type='bbox', all_masks=None, extra_path=''):
        """ detections_val2014_results.json """
        res_folder = os.path.join(self.result_path + extra_path, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, 'detections_%s_results.json' % self.image_set)
        self._write_coco_results(detections, res_file, ann_type, all_masks)
        if 'test' not in self.image_set:
            info_str = self._do_python_eval(res_file, res_folder, ann_type)
            return info_str

    def evaluate_sds(self, all_boxes, all_masks):
        info_str = self.evaluate_detections(all_boxes, 'segm', all_masks)
        return info_str

    def _write_coco_results(self, all_boxes, res_file, ann_type, all_masks):
        """ example results
        [{"image_id": 42,
          "category_id": 18,
          "bbox": [258.15,41.29,348.26,243.78],
          "score": 0.236}, ...]
        """
        all_im_info = [{'index': index,
                        'height': self.coco.loadImgs(index)[0]['height'],
                        'width': self.coco.loadImgs(index)[0]['width']}
                        for index in self.image_set_index]

        if ann_type == 'bbox':
            data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                          'cls_ind': cls_ind,
                          'cls': cls,
                          'ann_type': ann_type,
                          'binary_thresh': self.binary_thresh,
                          'all_im_info': all_im_info,
                          'boxes': all_boxes[cls_ind]}
                         for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']
        elif ann_type == 'segm':
            data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                          'cls_ind': cls_ind,
                          'cls': cls,
                          'ann_type': ann_type,
                          'binary_thresh': self.binary_thresh,
                          'all_im_info': all_im_info,
                          'boxes': all_boxes[cls_ind],
                          'masks': all_masks[cls_ind]}
                         for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']
        else:
            print 'unimplemented ann_type: '+ann_type
        # results = coco_results_one_category_kernel(data_pack[1])
        # print results[0]
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(coco_results_one_category_kernel, data_pack)
        pool.close()
        pool.join()
        results = sum(results, [])
        print 'Writing results json to %s' % res_file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _do_python_eval(self, res_file, res_folder, ann_type):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        info_str = self._print_detection_metrics(coco_eval)

        eval_file = os.path.join(res_folder, 'detections_%s_results.pkl' % self.image_set)
        with open(eval_file, 'w') as f:
            cPickle.dump(coco_eval, f, cPickle.HIGHEST_PROTOCOL)
        print 'coco eval results saved to %s' % eval_file
        info_str +=  'coco eval results saved to %s\n' % eval_file
        return info_str

    def _print_detection_metrics(self, coco_eval):
        info_str = ''
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)

        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print '~~~~ Mean and per-category AP @ IoU=%.2f,%.2f] ~~~~' % (IoU_lo_thresh, IoU_hi_thresh)
        info_str += '~~~~ Mean and per-category AP @ IoU=%.2f,%.2f] ~~~~\n' % (IoU_lo_thresh, IoU_hi_thresh)
        print '%-15s %5.1f' % ('all', 100 * ap_default)
        info_str += '%-15s %5.1f\n' % ('all', 100 * ap_default)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print '%-15s %5.1f' % (cls, 100 * ap)
            info_str +=  '%-15s %5.1f\n' % (cls, 100 * ap)

        print '~~~~ Summary metrics ~~~~'
        coco_eval.summarize()

        return info_str
