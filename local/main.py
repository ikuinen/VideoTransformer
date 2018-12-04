from dataloader import Loader
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
import criteria


# best, 2, 3, 512
class Locator(nn.Module):
    def __init__(self, d_model, frame_dim, word_dim):
        super().__init__()

        encoders = nn.ModuleList([
            EncoderLayer([config['max_words'], d_model],
                         MultiHeadAttention(num_heads, d_model, element_width=5),
                         PositionWiseFeedForward(d_model, d_ff)),
            EncoderLayer([config['max_words'], d_model],
                         MultiHeadAttention(num_heads, d_model),
                         PositionWiseFeedForward(d_model, d_ff)),
        ])
        self.encoder = MyEncoder(encoders)

        decoders = nn.ModuleList([
            DecoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(num_heads, d_model, element_width=5),
                         MultiHeadAttention(num_heads, d_model),
                         PositionWiseFeedForward(d_model, d_ff)),
            DecoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(num_heads, d_model, element_width=5),
                         MultiHeadAttention(num_heads, d_model),
                         PositionWiseFeedForward(d_model, d_ff)),
            DecoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(num_heads, d_model),
                         MultiHeadAttention(num_heads, d_model),
                         PositionWiseFeedForward(d_model, d_ff))
        ])

        self.decoder = MyDecoder(decoders)

        self.i_linear = nn.ModuleList([nn.Linear(frame_dim, d_model), nn.Linear(word_dim, d_model)])
        self.o_linear = clones(nn.Linear(d_model, 1), 2)
        self.pos1 = PositionEncoder(d_model, config['max_frames'])
        self.pos2 = PositionEncoder(d_model, config['max_words'])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, src, src_mask, tgt, tgt_mask, starts=None, ends=None):
        src = self.pos2(self.i_linear[1](src))
        tgt = self.pos1(self.i_linear[0](tgt))
        x = self.encoder(src, src_mask)  # [nb, len1, hid]

        x = self.decoder(tgt, x, src_mask, tgt_mask)  # [nb, len2, hid]
        start_logits = self.o_linear[0](x).view(-1, config['max_frames'])  # [nb, len]
        end_logits = self.o_linear[1](x).view(-1, config['max_frames'])  # [nb, len]

        if starts is not None:
            loss = self.loss(start_logits, starts) + self.loss(end_logits, ends)
            return F.softmax(start_logits, -1), F.softmax(end_logits, -1), loss
        return F.softmax(start_logits, -1), F.softmax(end_logits, -1)


if __name__ == '__main__':
    config = {
        "learning_rate": 5e-4,
        "lr_decay_n_iters": 3000,
        "max_epoches": 20,
        "early_stopping": 5,
        "display_batch_interval": 50,

        "dropout_prob": 0.9,

        "batch_size": 64,
        "d_model": 256,
        "d_ff": 256,
        "num_heads": 8,

        "input_video_dim": 500,
        "max_frames": 200,
        "input_ques_dim": 300,
        "max_words": 20,

        "word2vec": "data/word2vec.bin",

        "is_origin_dataset": True,
        "train_json": "data/activity-net/train.json",
        "val_json": "data/activity-net/val_1.json",
        "test_json": "data/activity-net/val_2.json",
        "train_data": "data/activity-net/train_data.json",
        "val_data": "data/activity-net/val_data.json",
        "test_data": "data/activity-net/test_data.json",
        "feature_path": "data/activity-c3d",
        "feature_path_tsn": "data/tsn_score"
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    word2vec = KeyedVectors.load_word2vec_format(config["word2vec"], binary=True)
    train_dataset = Loader(config, config['train_data'], word2vec, flag=True)
    val_dataset = Loader(config, config['val_data'], word2vec)
    test_dataset = Loader(config, config['test_data'], word2vec)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    d_model = config['d_model']
    d_ff = config['d_ff']
    num_heads = config['num_heads']

    locator = Locator(d_model=d_model,
                      frame_dim=config['input_video_dim'],
                      word_dim=config['input_ques_dim']).to(device)

    cnt = 0
    for name, para in locator.named_parameters():
        l = 1
        for j in para.size():
            l *= j
        cnt += l
    print('total variables: %d' % cnt)

    optimizer = torch.optim.Adam(locator.parameters(), lr=config['learning_rate'])

    def adjust_learning_rate(decay_rate=0.8):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    for epoch in range(1, config['max_epoches'] + 1):
        t1 = time.time()
        loss_sum = 0
        locator.train()
        for i_batch, (frame_vecs, frame_masks, ques_vecs, ques_masks, starts, ends) in enumerate(train_loader):
            frame_vecs = frame_vecs.to(device)
            frame_masks = frame_masks.to(device)
            ques_vecs = ques_vecs.to(device)
            ques_masks = ques_masks.to(device)
            starts = starts.to(device)
            ends = ends.to(device)

            _, _, loss = locator(ques_vecs, ques_masks, frame_vecs, frame_masks, starts, ends)

            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_value_(locator.parameters(), 3)
            optimizer.step()

            if i_batch % config['display_batch_interval'] == 0 and i_batch != 0:
                t2 = time.time()
                #for name, para in locator.named_parameters():
                #    if 'encoder.layers.0' in name:
                #        print(name, para.grad)
                print('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch' % (
                    epoch, i_batch, loss_sum / i_batch, (t2 - t1) / config['display_batch_interval']
                ))
                t1 = t2

        def eval(data_loader):
            total_IoU = 0.0
            all_retrievd = 0
            IoU0_1 = 0
            IoU0_3 = 0
            IoU0_5 = 0
            IoU0_7 = 0
            IoU0_9 = 0

            loss_sum = 0
            num_batches = 0
            locator.eval()
            for i_batch, (frame_vecs, frame_masks, ques_vecs, ques_masks, starts, ends) in enumerate(data_loader):
                frame_vecs = frame_vecs.to(device)
                frame_masks = frame_masks.to(device)
                ques_vecs = ques_vecs.to(device)
                ques_masks = ques_masks.to(device)
                starts = starts.to(device)
                ends = ends.to(device)

                start_probs, end_probs, loss = \
                    locator(ques_vecs, ques_masks, frame_vecs, frame_masks, starts, ends)
                loss_sum += loss.item()
                num_batches += 1
                start_probs = start_probs.cpu().detach().numpy()
                end_probs = end_probs.cpu().detach().numpy()

                # assert len(start_predicts) == len(starts)

                for s_vecs, e_vecs, s2, e2 in zip(start_probs, end_probs, starts.cpu().numpy(), ends.cpu().numpy()):
                    s1 = 0
                    e1 = 0
                    # print(s_vecs)
                    # print(e_vecs)
                    best = -1
                    for i in range(len(s_vecs)):
                        for j in range(i, len(e_vecs)):
                            if s_vecs[i] * e_vecs[j] > best:
                                best = s_vecs[i] * e_vecs[j]
                                s1 = i
                                e1 = j
                    # print((s1, e1), (s2, e2))
                    result = criteria.calculate_IoU((s1, e1), (s2, e2))
                    all_retrievd += 1
                    total_IoU += result
                    if result >= 0.1:
                        IoU0_1 += 1
                    if result >= 0.3:
                        IoU0_3 += 1
                    if result >= 0.5:
                        IoU0_5 += 1
                    if result >= 0.7:
                        IoU0_7 += 1
                    if result >= 0.9:
                        IoU0_9 += 1

                #if i_batch % config['display_batch_interval'] == 0 and i_batch != 0:
                #    t2 = time.time()
                    #print('Epoch %d, avg IoU %.4f, %.3f seconds/batch' % (epoch,
                    #                                                      total_IoU / all_retrievd,
                    #                                                      (t2 - t1) / config['display_batch_interval']))
                    #t1 = t2
            print('Epoch %d, mIoU %.4f, avg loss %.4f' % (
                epoch, total_IoU / all_retrievd, loss_sum / num_batches))
            print('Epoch %d, IoU@0.1 %.4f' % (epoch, IoU0_1 / all_retrievd))
            print('Epoch %d, IoU@0.3 %.4f' % (epoch, IoU0_3 / all_retrievd))
            print('Epoch %d, IoU@0.5 %.4f' % (epoch, IoU0_5 / all_retrievd))
            print('Epoch %d, IoU@0.7 %.4f' % (epoch, IoU0_7 / all_retrievd))
            print('Epoch %d, IoU@0.9 %.4f' % (epoch, IoU0_9 / all_retrievd))


        print('Epoch %d, eval set:' % epoch)
        eval(val_loader)
        print('Epoch %d, test set:' % epoch)
        eval(test_loader)

        if epoch % 5 == 0:
            adjust_learning_rate(0.5)