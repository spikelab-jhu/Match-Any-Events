from yacs.config import CfgNode as CN
_CN = CN()

_CN.MAE = CN()
_CN.MAE.RESOLUTION = (14, 1)
_CN.NUM_BINS = 8
_CN.RES = (350, 630)
_CN.MAE.FINE_WINDOW_SIZE = 14  
_CN.MAE.MP = True
_CN.MAE.REPLACE_NAN = True
_CN.MAE.EVAL_TIMES = 1
_CN.MAE.HALF = True

_CN.MAE.FINETUNE = False
_CN.MAE.ARCHITECTURE = 'TAg'
_CN.MAE.TRAIN_DATA = 'default'

_CN.MAE.EST = CN()
_CN.MAE.EST.D_MODEL = 384
_CN.MAE.EST.NHEAD = 4
_CN.MAE.EST.SEQ_LEN = 8
_CN.MAE.EST.MP = True

_CN.MAE.EST.HALF = True
_CN.MAE.EST.LAYER_NAMES = ['spatial', 'temporal'] * 2
_CN.MAE.EST.AGG_SIZE0 = 1
_CN.MAE.EST.AGG_SIZE1 = 1
_CN.MAE.EST.NO_FLASH = False
_CN.MAE.EST.ROPE = True
_CN.MAE.EST.NPE = [350, 630, 350, 630]
_CN.MAE.EST.T_NPE = [8, 8]

_CN.MAE.TEMPORAL_PHASE = True

_CN.MAE.MUTRA = CN()
_CN.MAE.MUTRA.LEARNED_BIAS = True
_CN.MAE.MUTRA.LAYERS = 4

_CN.MAE.MUTRA.EMBED_DIM = 384
_CN.MAE.MUTRA.NUM_HEADS = 4
_CN.MAE.MUTRA.DROPOUT = 0.0
_CN.MAE.MUTRA.ATTN_DROPOUT = 0.


_CN.MAE.DINO = CN()
_CN.MAE.DINO.MODEL = 's' # in ['b', 's']
_CN.MAE.DINO.IMG_SIZE = 630
_CN.MAE.DINO.IN_CHAN = 2
_CN.MAE.DINO.PATCH_SIZE = 14
_CN.MAE.DINO.DIM = 384

# 2. MAE-coarse module config
_CN.MAE.COARSE = CN()
_CN.MAE.COARSE.D_MODEL = 256
_CN.MAE.COARSE.D_FFN = 256
_CN.MAE.COARSE.NHEAD = 8
_CN.MAE.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.MAE.COARSE.AGG_SIZE0 = 1
_CN.MAE.COARSE.AGG_SIZE1 = 1
_CN.MAE.COARSE.NO_FLASH = False
_CN.MAE.COARSE.ROPE = True
_CN.MAE.COARSE.NPE = [350, 630, 350, 630] # [832, 832, long_side, long_side] Suggest setting based on the long side of the input image, especially when the long_side > 832

# 3. Coarse-Matching config
_CN.MAE.MATCH_COARSE = CN()
_CN.MAE.MATCH_COARSE.THR = 0.2# recommend 0.2 for full model and 25 for optimized model
_CN.MAE.MATCH_COARSE.BORDER_RM = 2
_CN.MAE.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MAE.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3  # training tricks: save GPU memory
_CN.MAE.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 20  # training tricks: avoid DDP deadlock
_CN.MAE.MATCH_COARSE.SPARSE_SPVS = True
_CN.MAE.MATCH_COARSE.SKIP_SOFTMAX = False
_CN.MAE.MATCH_COARSE.FP16MATMUL = False

# 4. Fine-Matching config
_CN.MAE.MATCH_FINE = CN()
_CN.MAE.MATCH_FINE.SPARSE_SPVS = True
_CN.MAE.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0
_CN.MAE.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8

# 5. MAE Losses
# -- # coarse-level
_CN.MAE.LOSS = CN()
_CN.MAE.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.MAE.LOSS.COARSE_WEIGHT = 1.0
_CN.MAE.LOSS.COARSE_SIGMOID_WEIGHT = 1.0
_CN.MAE.LOSS.LOCAL_WEIGHT = 0.25
_CN.MAE.LOSS.TOKEN_WEIGHT = 3.0
_CN.MAE.LOSS.POUNDER_WEIGHT = 9e-3
_CN.MAE.LOSS.CERTAINTY_WEIGHT = 0
_CN.MAE.LOSS.CONTRAST_WEIGHT = 0.0
_CN.MAE.LOSS.COARSE_OVERLAP_WEIGHT = False
_CN.MAE.LOSS.FINE_OVERLAP_WEIGHT = False
_CN.MAE.LOSS.FINE_OVERLAP_WEIGHT2 = False
# -- - -- # focal loss (coarse)
_CN.MAE.LOSS.FOCAL_ALPHA = 0.25
_CN.MAE.LOSS.FOCAL_GAMMA = 2.0
_CN.MAE.LOSS.POS_WEIGHT = 1.0
_CN.MAE.LOSS.NEG_WEIGHT = 1.0

# -- # fine-level
_CN.MAE.LOSS.FINE_TYPE = 'l2'  # ['l2_with_std', 'l2']
_CN.MAE.LOSS.FINE_WEIGHT = 1.0
_CN.MAE.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)


##############  Dataset  ##############
_CN.DATASET = CN()
# dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.4

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-4
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 1000

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [4,8,12,16]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, LO-RANSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
