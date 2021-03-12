import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dataset import *
from model import *
import h5py
from transformers import create_optimizer
from sklearn.metrics import f1_score, accuracy_score
import time
from datetime import timedelta
from config import cfg
from tensorflow.keras.mixed_precision import experimental as mixed_precision

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())

@tf.function
def train_step_ad(input, optimizers, index, model):
    """
    使用不同学习率
    :param text:
    :param tag:
    :param mask:
    :param optimizer:
    :param model:
    :return:
    """
    """
    1. 计算loss对em的倒数
    2. 更新em
    3. 再次计算
    4. 跟新
    """
    # 打开梯度记录管理器
    with tf.GradientTape() as tape:
        loss = model.get_loss(input)
        scaled_loss = optimizers.get_scaled_loss(loss)

    grads = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizers.get_unscaled_gradients(grads)
    # 使用梯度和优化器更新参数
    # optimizers[0].apply_gradients(zip(gradients[:index], model.trainable_variables[:index]))
    # optimizers[1].apply_gradients(zip(gradients[index:], model.trainable_variables[index:]))
    optimizers.apply_gradients(zip(gradients, model.trainable_variables))
    # 返回平均损失
    return loss

def dev_step(data, model, dev_count):
    start = time.time()
    p_all = []
    tag = []
    tqdm_ = tqdm(total=dev_count)
    for _ in data:
        input_ids, token_type_ids, attention_mask, sep_index, label = _
        input = [input_ids, token_type_ids, attention_mask, sep_index]
        p = model.predict(input)
        p_all.extend(p)
        tag.extend(label.numpy().tolist())
        tqdm_.update(1)

    f1 = f1_score(tag, p_all, average='macro')
    # f1 = accuracy_score(tag, p_all)
    end = time.time()
    tqdm_.close()
    return f1, end-start

def train_and_test(model, train, dev, epoch, count, dev_count, optimizers, index, ckpt_manager):
    total_loss = []
    iter_ = 0
    # tqdm_ = tqdm(total=epoch*count)
    start = time.time()
    best_f1 = 0
    for e in range(epoch):
        cache = []
        for _ in train:
            # if len(cache)<=2:
            #     cache.append(_)
            #     continue
            loss = train_step_ad(_, optimizers, index, model)
            total_loss.append(loss)
            iter_+=1
            # tqdm_.update(1)

            if iter_%500==0:
                f1,t = dev_step(dev, model, dev_count)
                print('Epoch {} step {}/{} loss: {:.4f} f1: {:.4f} {}'.format(e + 1, iter_, count*epoch, sum(total_loss)/len(total_loss), f1, str(timedelta(seconds=int(t)))))
                total_loss = []
                if f1>best_f1:
                    best_f1 = f1
                    ckpt_manager.save()

            elif iter_%100==0:
                end = time.time()
                print('Epoch {} step {}/{} loss: {:.4f} {}'.format(e + 1, iter_, count*epoch, sum(total_loss)/len(total_loss), str(timedelta(seconds=int(end-start)))))
                total_loss = []
                # ckpt_manager.save()
                start = end
    # tqdm_.close()

if __name__=='__main__':
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    save_f = 'test'
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    train = Dataset(read_json(cfg.TRAIN), batch_size= cfg.BATCH_SIZE)
    count = train.total
    train = get_dataset_by_iter(train)
    train = get_data(train)

    dev = Dataset(read_json(cfg.DEV), batch_size= cfg.BATCH_SIZE)
    dev_count = dev.total
    dev = get_dataset_by_iter(dev)
    dev = get_data(dev, shuffle=0)

    model = RL()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              './model/'+save_f,
                                              checkpoint_name='model.ckpt',
                                              max_to_keep=1)

    for _ in train:
        init_call = _
        break

    model.get_loss(init_call)
    for i, x in enumerate(model.trainable_variables):
        print("%d:%s %s" % (i, x.name, str(x.shape)))

    EPOCHS = cfg.EPOCHS
    optimizer2,_ = create_optimizer(2e-5, count * EPOCHS, count, 0, weight_decay_rate=0.01)
    optimizer2 = mixed_precision.LossScaleOptimizer(optimizer2, loss_scale='dynamic')

    train_and_test(model, train, dev, EPOCHS, count, dev_count, optimizer2, 197, ckpt_manager)