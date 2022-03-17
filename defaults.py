# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------------------- #
#                           Input                          #
# -------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.DATASET = "CUHK-SYSU"
_C.INPUT.DATA_ROOT = "data/CUHK-SYSU"

# Size of the smallest side of the image
_C.INPUT.MIN_SIZE = 900
# Maximum size of the side of the image
_C.INPUT.MAX_SIZE = 1500

# Number of images per batch
_C.INPUT.BATCH_SIZE_TRAIN = 5
_C.INPUT.BATCH_SIZE_TEST = 1

# Number of data loading threads
_C.INPUT.NUM_WORKERS_TRAIN = 5
_C.INPUT.NUM_WORKERS_TEST = 1

# Image augmentation
_C.INPUT.IMAGE_CUTOUT = False
_C.INPUT.IMAGE_ERASE = False
_C.INPUT.IMAGE_MIXUP = False

# -------------------------------------------------------- #
#                           GRID                           #
# -------------------------------------------------------- #
_C.INPUT.IMAGE_GRID = False
_C.GRID = CN()
_C.GRID.ROTATE = 1
_C.GRID.OFFSET = 0
_C.GRID.RATIO = 0.5
_C.GRID.MODE = 1
_C.GRID.PROB = 0.5

# -------------------------------------------------------- #
#                          Solver                          #
# -------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 13

# Learning rate settings
_C.SOLVER.BASE_LR = 0.003

# The epoch milestones to decrease the learning rate by GAMMA
_C.SOLVER.LR_DECAY_MILESTONES = [10, 14]
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.SGD_MOMENTUM = 0.9

# Loss weight of RPN regression
_C.SOLVER.LW_RPN_REG = 1
# Loss weight of RPN classification
_C.SOLVER.LW_RPN_CLS = 1

# Loss weight of Cascade R-CNN and Re-ID (OIM)
_C.SOLVER.LW_RCNN_REG_1ST = 10
_C.SOLVER.LW_RCNN_CLS_1ST = 1
_C.SOLVER.LW_RCNN_REG_2ND = 10
_C.SOLVER.LW_RCNN_CLS_2ND = 1
_C.SOLVER.LW_RCNN_REG_3RD = 10
_C.SOLVER.LW_RCNN_CLS_3RD = 1
_C.SOLVER.LW_RCNN_REID_2ND = 0.5
_C.SOLVER.LW_RCNN_REID_3RD = 0.5
# Loss weight of box reid, softmax loss
_C.SOLVER.LW_RCNN_SOFTMAX_2ND = 0.5
_C.SOLVER.LW_RCNN_SOFTMAX_3RD = 0.5

# Set to negative value to disable gradient clipping
_C.SOLVER.CLIP_GRADIENTS = 10.0

# -------------------------------------------------------- #
#                            RPN                           #
# -------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.RPN = CN()
# NMS threshold used on RoIs
_C.MODEL.RPN.NMS_THRESH = 0.7
# Number of anchors per image used to train RPN
_C.MODEL.RPN.BATCH_SIZE_TRAIN = 256
# Target fraction of foreground examples per RPN minibatch
_C.MODEL.RPN.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an anchor to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.RPN.POS_THRESH_TRAIN = 0.7
# Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.RPN.NEG_THRESH_TRAIN = 0.3
# Number of top scoring RPN RoIs to keep before applying NMS
_C.MODEL.RPN.PRE_NMS_TOPN_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPN_TEST = 6000
# Number of top scoring RPN RoIs to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOPN_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPN_TEST = 300

# -------------------------------------------------------- #
#                         RoI head                         #
# -------------------------------------------------------- #
_C.MODEL.ROI_HEAD = CN()
# Whether to use bn neck (i.e. batch normalization after linear)
_C.MODEL.ROI_HEAD.BN_NECK = True
# Number of RoIs per image used to train RoI head
_C.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN = 128
# Target fraction of foreground examples per RoI minibatch
_C.MODEL.ROI_HEAD.POS_FRAC_TRAIN = 0.25 # 0.5

_C.MODEL.ROI_HEAD.USE_DIFF_THRESH = True
# Overlap threshold for an RoI to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN = 0.5
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND = 0.6
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN_3RD = 0.7
# Overlap threshold for an RoI to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN = 0.5
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND = 0.6
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_3RD = 0.7
# Minimum score threshold
_C.MODEL.ROI_HEAD.SCORE_THRESH_TEST = 0.5
# NMS threshold used on boxes
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST = 0.4
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST_1ST = 0.4
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND = 0.4
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST_3RD = 0.5
# Maximum number of detected objects
_C.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST = 300

# -------------------------------------------------------- #
#                     Transformer head                     #
# -------------------------------------------------------- #
_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.DIM_MODEL = 512
_C.MODEL.TRANSFORMER.ENCODER_LAYERS = 1
_C.MODEL.TRANSFORMER.N_HEAD = 8
_C.MODEL.TRANSFORMER.USE_OUTPUT_LAYER = False
_C.MODEL.TRANSFORMER.DROPOUT = 0.
_C.MODEL.TRANSFORMER.USE_LOCAL_SHORTCUT = True
_C.MODEL.TRANSFORMER.USE_GLOBAL_SHORTCUT = True

_C.MODEL.TRANSFORMER.USE_DIFF_SCALE = True
_C.MODEL.TRANSFORMER.NAMES_1ST = ['scale1','scale2']
_C.MODEL.TRANSFORMER.NAMES_2ND = ['scale1','scale2']
_C.MODEL.TRANSFORMER.NAMES_3RD = ['scale1','scale2']
_C.MODEL.TRANSFORMER.KERNEL_SIZE_1ST = [(1,1),(3,3)]
_C.MODEL.TRANSFORMER.KERNEL_SIZE_2ND = [(1,1),(3,3)]
_C.MODEL.TRANSFORMER.KERNEL_SIZE_3RD = [(1,1),(3,3)]
_C.MODEL.TRANSFORMER.USE_MASK_1ST = False
_C.MODEL.TRANSFORMER.USE_MASK_2ND = True
_C.MODEL.TRANSFORMER.USE_MASK_3RD = True
_C.MODEL.TRANSFORMER.USE_PATCH2VEC = True

####
_C.MODEL.USE_FEATURE_MASK = True
_C.MODEL.FEATURE_AUG_TYPE = 'exchange_token' # 'exchange_token', 'jigsaw_token', 'cutout_patch', 'erase_patch', 'mixup_patch', 'jigsaw_patch'
_C.MODEL.FEATURE_MASK_SIZE = 4
_C.MODEL.MASK_SHAPE = 'stripe' # 'square', 'random'
_C.MODEL.MASK_SIZE = 1
_C.MODEL.MASK_MODE = 'random_direction' # 'horizontal', 'vertical' for stripe; 'random_size' for square
_C.MODEL.MASK_PERCENT = 0.1
#### 
_C.MODEL.EMBEDDING_DIM = 256

# -------------------------------------------------------- #
#                           Loss                           #
# -------------------------------------------------------- #
_C.MODEL.LOSS = CN()
# Size of the lookup table in OIM
_C.MODEL.LOSS.LUT_SIZE = 5532
# Size of the circular queue in OIM
_C.MODEL.LOSS.CQ_SIZE = 5000
_C.MODEL.LOSS.OIM_MOMENTUM = 0.5
_C.MODEL.LOSS.OIM_SCALAR = 30.0

_C.MODEL.LOSS.USE_SOFTMAX = True

# -------------------------------------------------------- #
#                        Evaluation                        #
# -------------------------------------------------------- #
# The period to evaluate the model during training
_C.EVAL_PERIOD = 1
# Evaluation with GT boxes to verify the upper bound of person search performance
_C.EVAL_USE_GT = False
# Fast evaluation with cached features
_C.EVAL_USE_CACHE = False
# Evaluation with Context Bipartite Graph Matching (CBGM) algorithm
_C.EVAL_USE_CBGM = False
# Gallery size in evaluation, only for CUHK-SYSU
_C.EVAL_GALLERY_SIZE = 100
# Feature used for evaluation
_C.EVAL_FEATURE = 'concat' # 'stage2', 'stage3'

# -------------------------------------------------------- #
#                           Miscs                          #
# -------------------------------------------------------- #
# Save a checkpoint after every this number of epochs
_C.CKPT_PERIOD = 1
# The period (in terms of iterations) to display training losses
_C.DISP_PERIOD = 10
# Whether to use tensorboard for visualization
_C.TF_BOARD = True
# The device loading the model
_C.DEVICE = "cuda"
# Set seed to negative to fully randomize everything
_C.SEED = 1
# Directory where output files are written
_C.OUTPUT_DIR = "./output"


def get_default_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
