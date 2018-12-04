import argparse
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()

Epoch = namedtuple('Epoch', ['mIoU', 'IoU0_1', 'IoU0_3', 'IoU0_5', 'IoU0_7', 'IoU0_9'])
val = True
mIoU, IoU0_1, IoU0_3, IoU0_5, IoU0_7, IoU0_9 = None, None, None, None, None, None
vals = []
tests = []
with open(args.file, encoding='utf8') as fp:
    for line in fp.readlines():
        if 'IoU' in line:
            i = line.rstrip().split()
            if i[2] == 'mIoU':
                mIoU = float(i[3][:-1])
            elif i[2] == 'IoU@0.1':
                IoU0_1 = float(i[3])
            elif i[2] == 'IoU@0.3':
                IoU0_3 = float(i[3])
            elif i[2] == 'IoU@0.5':
                IoU0_5 = float(i[3])
            elif i[2] == 'IoU@0.7':
                IoU0_7 = float(i[3])
            elif i[2] == 'IoU@0.9':
                IoU0_9 = float(i[3])
                epoch = Epoch(mIoU, IoU0_1, IoU0_3, IoU0_5, IoU0_7, IoU0_9)
                if val:
                    val = False
                    vals.append(epoch)
                else:
                    val = True
                    tests.append(epoch)

def show(epochs):
    max_i = None
    for i, epoch in enumerate(epochs):
        if max_i is None or epoch.mIoU > epochs[max_i].mIoU:
            max_i = i
    #print('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (max_i, epochs[max_i].mIoU, epochs[max_i].IoU0_1, epochs[max_i].IoU0_3,
    #                                  epochs[max_i].IoU0_5, epochs[max_i].IoU0_7, epochs[max_i].IoU0_9))
    print(max_i, epochs[max_i])


show(vals)
show(tests)
