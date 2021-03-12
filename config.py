from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C
__C.DIR                       = './model/bert-base-chinese'
__C.HIDDENT_SIZE              = 768
__C.DROPOUT                   = 0.5
__C.MAX_LENGTH                = 128
__C.BATCH_SIZE                = 8
__C.EPOCHS                    = 10
__C.GPU                       = '1'
__C.TRAIN                     = './data/train_p.json'
__C.DEV                       = './data/dev_p.json'
