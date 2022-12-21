# Y. Feng, Z. Zhang, X. Zhao, R. Ji, and Y. Gao, “GVCNN: Group-view convolutional neural networks for 3D shape recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jun. 2018, pp. 264–272
import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils.models import inceptionV4


class GVCNN(nn.Module): # input 3*224*224
    def __init__(self, nclasses, num_views):
        super(GVCNN, self).__init__()

        self.nclasses = nclasses
        self.num_views = num_views

        self.net = inceptionV4.inceptionv4()
        self.net.last_linear = nn.Linear(1536, nclasses)
        self.net.last_linear.apply(init_weights)

        # first six layers of inception_v4
        self.fcn_1 = nn.Sequential(*list(self.net.features[0:5]))

        # grouping module
        self.group_schema = GroupSchema()
        init_weights(self.group_schema)

        # remain layers of inception_v4
        self.fcn_2 = nn.Sequential(*list(self.net.features[5:]))

        self.avg_pool_2 = nn.AvgPool2d(5, count_include_pad=False) # self.net.avg_pool
        self.fc_2 = self.net.last_linear

    def forward(self, inputs):
        imgs, doppiaAngolazione = inputs
        
        """
        :param x: Batch View Channel Height Width
        :return:
        """
        x = torch.cat((imgs.unsqueeze(1), doppiaAngolazione.unsqueeze(1)), dim=1)

        # transform the x from [N V C H W] to [N*V C H W]
        x = x.view((int(x.shape[0] * self.num_views), x.shape[-3], x.shape[-2], x.shape[-1]))
        # print('[24 3 224 224]', x.size())

        # [N*V 192 52 52]
        y = self.fcn_1(x)
        # print('[24 192 52 52]', y.size())

        # [N V 192 52 52]
        y1 = y.view(
            (int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))  # (8,12,512,7,7)
        # print('[2 12 192 52 52]', y1.size())

        # [V N 192 52 52]
        raw_view = y1.transpose(0, 1)
        # print('[12, 2, 192, 52, 52]', raw_view.size())

        # [N V] scores
        view_scores = self.group_schema(raw_view)
        # print('[2 12]', view_scores.size())

        # [NV 1536 5 5]
        final_view = self.fcn_2(y)
        # print('[24 1536 5 5]', final_view.size())

        # [N V C H W]
        final_view = final_view.view(
            (int(final_view.shape[0]/self.num_views)),
            self.num_views, final_view.shape[-3],
            final_view.shape[-2], final_view.shape[-1]
        )
        # print('[2 12 1536 5 5]', final_view.size())

        # [N C H W]
        shape_decriptors = group_pool(final_view, view_scores)
        z = self.avg_pool_2(shape_decriptors)
        z = z.view(z.size(0), -1)
        # [N num_classes]
        z = self.fc_2(z)
        return z


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class OneConvFc(nn.Module):
    """
    1*1 conv + fc to obtain the grouping schema
    """
    def __init__(self):
        super(OneConvFc, self).__init__()
        self.conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, stride=1)
        self.fc = nn.Linear(in_features=52*52, out_features=1)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GroupSchema(nn.Module):
    """
    differences from paper:
    1. Considering the amount of params, we use 1*1 conv instead of  fc
    2. if the scores are all very small, it will cause a big problem in params' update,
    so we add a softmax layer to normalize the scores after the convolution layer
    """
    def __init__(self):
        super(GroupSchema, self).__init__()
        self.score_layer = OneConvFc()
        self.sft = nn.Softmax(dim=1)

    def forward(self, raw_view):
        """
        :param raw_view: [V N C H W]
        :return:
        """
        scores = []
        for batch_view in raw_view:
            # batch_view: [N C H W]
            # y: [N]
            y = self.score_layer(batch_view)
            y = torch.sigmoid(torch.log(torch.abs(y)))
            scores.append(y)
        # view_scores: [N V]
        view_scores = torch.stack(scores, dim=0).transpose(0, 1)
        view_scores = view_scores.squeeze(dim=-1)
        return self.sft(view_scores)


def view_pool(ungrp_views, view_scores, num_grps=7):
    """
    :param ungrp_views: [V C H W]
    :param view_scores: [V]
    :param num_grps the num of groups. used to calc the interval of each group.
    :return: grp descriptors [(grp_descriptor, weight)]
    """

    def calc_scores(scores):
        """
        :param scores: [score1, score2 ....]
        :return:
        """
        n = len(scores)
        s = torch.ceil(scores[0]*n)
        for idx, score in enumerate(scores):
            if idx == 0:
                continue
            s += torch.ceil(score*n)
        s /= n
        return s

    interval = 1 / (num_grps + 1)
    # begin = 0
    view_grps = [[] for i in range(num_grps)]
    score_grps = [[] for i in range(num_grps)]

    for idx, (view, view_score) in enumerate(zip(ungrp_views, view_scores)):
        begin = 0
        for j in range(num_grps):
            right = begin + interval
            if j == num_grps-1:
                right = 1.1
            if begin <= view_score < right:
                view_grps[j].append(view)
                score_grps[j].append(view_score)
            begin += interval
    # print(score_grps)
    view_grps = [sum(views)/len(views) for views in view_grps if len(views) > 0]
    score_grps = [calc_scores(scores) for scores in score_grps if len(scores) > 0]

    shape_des = map(lambda a, b: a*b, view_grps, score_grps)
    shape_des = sum(shape_des)/sum(score_grps)

    # !!! if all scores are very small, it will cause some problems in params' update
    if sum(score_grps) < 0.1:
        # shape_des = sum(view_grps)/len(score_grps)
        print(sum(score_grps), score_grps)
    # print('score total', score_grps)
    return shape_des


def group_pool(final_view, scores):
    """
    view pooling + group fusion
    :param final_view: # [N V C H W]
    :param scores: [N V] scores
    :return: shape descriptor
    """
    shape_descriptors = []

    for idx, (ungrp_views, view_scores) in enumerate(zip(final_view, scores)):
        # ungrp_views: [V C H W]
        # view_scores: [V]

        # view pooling
        shape_descriptors.append(view_pool(ungrp_views, view_scores))
    # [N C H W]
    y = torch.stack(shape_descriptors, 0)
    # print('[2 C H W]', y.size())
    return y