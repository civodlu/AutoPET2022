import trw

# different splits:
# https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md

# SBD (additional annotations for voc2012)
# https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal


# other impl https://github.com/Lextal/pspnet-pytorch
# https://github.com/speedinghzl/Pytorch-Deeplab

#
# TODO CHECK [good]
#  https://github.com/yassouali/pytorch_segmentation
#  https://github.com/hszhao/semseg
#

d = trw.datasets.create_voc_segmentation_dataset(batch_size=16)
for b in d['voc2012_seg']['train']:
    print(b)