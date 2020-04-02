import torch
import torchvision
import numpy as np


'''
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n,)
    pose_preds:     pose locations list (n, 17, 2)
    pose_scores:    pose scores list    (n, 17, 1)
    
    0       nose
    1       left_eye
    2       right_eye   
    3       left_ear
    4       right_ear
    5       left_shoulder
    6       right_shoulder
    7       left_elbow
    8       right_elbow
    9       left_wrist  
    10      right_wrist
    11      left_hip
    12      right_hip
    13      left_knee
    14      right_knee
    15      left_ankle
    16      right_ankle
    
    part_mask
    0       head
    1       shoulder
    2       left_elbow
    3       right_elbow
    4       left_arm
    5       right_arm
    6       hip
    7       body
    8       left_knee
    9       right_knee
    10      left_ankle
    11      right_ankle
'''
bboxes = torch.tensor([0,0,200,400])
one_kp_points=torch.tensor([[100,30],
                       [105,25],
                       [95,25],
                       [110,25],
                       [90,25],
                       [130,50],
                       [70,50],
                       [150,20],
                       [50,150],
                       [180,5],
                       [60,210],
                       [130,250],
                       [70,250],
                       [95,325],
                       [35,325],
                       [140,390],
                       [95,380]])
# one_kp_scores = torch.tensor([0.9, 0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.8, 0.7, 0.15, 0.25, 0.5, 0.5, 0.45, 0.45, 0.3, 0.1])
one_kp_scores = torch.tensor([0.9, 0.8, 0.8, 0.8, 0.8, 0.075, 0.75, 0.8, 0.7, 0.35, 0.25, 0.5, 0.5, 0.45, 0.45, 0.3, 0.3])
print(bboxes.shape)
print(one_kp_points.shape)
print(one_kp_scores.shape)


def get_part_point(kp_points,kp_scores):
    #分别计算各个部位点的坐标
    head_point = kp_points[0]
    shoulder_point = (kp_points[5] + kp_points[6]) / 2
    left_elbow_point = (kp_points[5] + kp_points[7]) / 2
    right_elbow_point = (kp_points[6] + kp_points[8]) / 2
    left_arm_point = (kp_points[7] + kp_points[9]) / 2
    right_arm_point = (kp_points[8] + kp_points[10]) / 2
    hip_point = (kp_points[11] + kp_points[12]) / 2
    body_point = (shoulder_point + hip_point) / 2
    left_knee_point = (kp_points[11] + kp_points[13]) / 2
    right_knee_point = (kp_points[12] + kp_points[14]) / 2
    left_ankle_point = (kp_points[13] + kp_points[15]) / 2
    right_ankle_point = (kp_points[14] + kp_points[16]) / 2

    part_points = [head_point,shoulder_point,left_elbow_point,right_elbow_point,left_arm_point,right_arm_point,
                   hip_point,body_point,left_knee_point,right_knee_point,left_ankle_point,right_ankle_point]

    score_mask = kp_scores>=0.2
    # print(score_mask)
    part_mask_init = [True,True,True,True,True,True,True,True,True,True,True,True]

    part_mask = get_part_mask(score_mask,part_mask_init)

    #返回部位点以及部位mask
    return part_points,part_mask

def get_part_mask(score_mask,part_mask):
    #check head_mask
    if sum(score_mask[0:5])==0:
        part_mask[0]=False
    '''shoulder'''
    # left shoulder = False,part of shoulder,left elbow, will be false
    if score_mask[5] == False:
        part_mask[1] = part_mask[2] = False

    # right shulder = False,part of shoulder,right elbow,body will be false
    if score_mask[6]==0:
        part_mask[1] = part_mask[3] = False

    '''arm'''
    # left elbow = False ,part of left arm
    if score_mask[7] == False:
        part_mask[2] = part_mask[4] = False
    # left wrist = false
    if score_mask[9] == False:
        part_mask[4] = False

    # right elbow = False ,part of right arm
    if score_mask[8] == False:
        part_mask[3] = part_mask[5] = False
    # right wrist =false
    if score_mask[10] == False:
        part_mask[5] = False

    '''hip'''
    # left hip = false,hip ,body ,left knee false
    if score_mask[11] == False:
        part_mask[6] =  part_mask[8] = False

    # right hip = false,hip ,body ,right knee false
    if score_mask[12] == False:
        part_mask[6] =  part_mask[9] = False

    #两个肩膀缺失或者两个臀部的都缺失导致其无法定位的时候body才会都缺失
    if not((score_mask[5]or score_mask[6])and(score_mask[5]or score_mask[6])):
        part_mask[7] = False

    '''leg'''
    # left knee =false ,part of left leg
    if score_mask[13] == False:
        part_mask[8] = part_mask[10] = False
    # left ankle = false
    if score_mask[15] == False:
        part_mask[10] = False

    # right knee =false ,part of right leg
    if score_mask[14] == False:
        part_mask[9] = part_mask[11] = False

    # right ankle = false
    if score_mask[16] == False:
        part_mask[11] = False
    # part_mask = torch.Tensor(part_mask)
    return part_mask

def get_part_distance(mask,bbox,part_point):
    x1, y1, x2, y2 = bboxes[0], bboxes[1], bboxes[2], bboxes[3]

    part_list = ['head','shoulder','left_elbow','right_elbow','left_arm','right_arm','hip','body',
                 'left_knee','right_knee','left_ankle','right_ankle']
    #定义dict用来存储各点到各边的距离
    top_dist, bottom_dist, left_dist, right_dist= {},{},{},{}
    #计算各点到各边的距离
    for i in range(12):
        top_dist['%s' % part_list[i]] = part_point[i][1] - y1
        bottom_dist['%s' % part_list[i]] = y2 - part_point[i][1]
        left_dist['%s' % part_list[i]] = part_point[i][0] - x1
        right_dist['%s' % part_list[i]] = x2 - part_point[i][0]
        # print('%s'%part_list[i],dist_top,dist_bottom,dist_left,dist_right)

    #delete occlused part 删除被遮挡的部位
    for i in range(12):
        if mask[i] == False:
            del(top_dist['%s'%part_list[i]],bottom_dist['%s' % part_list[i]],left_dist['%s' % part_list[i]],right_dist['%s' % part_list[i]])

    dists = (top_dist,bottom_dist,left_dist,right_dist)
    # return top_dist,bottom_dist,left_dist,right_dist

    return dists
def bbox_code(bbox,distances,kp_points):
    W = bbox[2]-bbox[0]
    H = bbox[3]-bbox[1]
    X0,Y0 = bbox[0],bbox[1]
    top,bottom,left,right = np.zeros(W),np.zeros(W),np.zeros(H),np.zeros(H)
    top_dist, bottom_dist, left_dist, right_dist = distances
    #对各个部位进行排序
    # top_dist=sorted(top_dist.items(),key=lambda item:item[1][0])
    # bottom_dist = sorted(bottom_dist.items(),key=lambda item:item[1][0])
    # left_dist = sorted(left_dist.items(), key=lambda item: item[1][0])
    # right_dist = sorted(right_dist.items(), key=lambda item: item[1][0])

    top_dist = sorted(top_dist.items(), key=lambda item: item[1])
    bottom_dist = sorted(bottom_dist.items(), key=lambda item: item[1])
    left_dist = sorted(left_dist.items(), key=lambda item: item[1])
    right_dist = sorted(right_dist.items(), key=lambda item: item[1])
    #存在的部位的长度
    part_len = len(top_dist)
    top = context_x(X0,top,kp_points,top_dist,part_len)
    bottom = context_x(X0,bottom,kp_points,bottom_dist,part_len)
    left = context_y(Y0, left, kp_points, left_dist, part_len)
    right = context_y(Y0, right, kp_points, right_dist, part_len)

    bbox_context = (top,bottom,left,right)
    return bbox_context

#对横向的边框进行编码
def context_x(X,line,kp_,sorted_dist,part_num):
    for i in range(part_num):
        if sorted_dist[i][0] == 'head':
            head_w = max(2*abs(kp_[1][0]-kp_[2][0]),abs(kp_[3][0]-kp_[4][0]))
            start = kp_[0][0]-int(head_w/2)-X
            for t in range(head_w+1):
                if line[start+t] == 0:
                    line[start+t]=1
        if sorted_dist[i][0] == 'shoulder':
            start = min(kp_[6][0],kp_[7][0]) - X
            len = abs(kp_[6][0] - kp_[7][0])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 2
        if sorted_dist[i][0] == 'left_elbow':
            start = min(kp_[5][0],kp_[7][0]) - X
            len = abs(kp_[5][0]-kp_[7][0])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 3
        if sorted_dist[i][0] == 'right_elbow':
            start = min(kp_[6][0],kp_[8][0]) - X
            len = abs(kp_[6][0]-kp_[8][0])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 4
        if sorted_dist[i][0] == 'left_arm':
            start = min(kp_[7][0],kp_[9][0]) - X
            len = abs(kp_[7][0]-kp_[9][0])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 5
        if sorted_dist[i][0] == 'right_arm':
            start = min(kp_[8][0],kp_[10][0]) - X
            len = abs(kp_[8][0]-kp_[10][0])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 6
        if sorted_dist[i][0] == 'hip':
            start = min(kp_[11][0],kp_[12][0]) - X
            len = abs(kp_[11][0] - kp_[12][0])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 7
        #身体的部分比较复杂，四个点确定
        if sorted_dist[i][0] == 'body':
            Xs = (kp_[5]+kp_[6])/2
            Xh = (kp_[11]+kp_[12])/2
            len = abs(Xs[0]-Xh[0])
            start = min(Xs[0],Xh[0])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 8
        if sorted_dist[i][0] == 'left_knee':
            start = min(kp_[11][0], kp_[13][0]) - X
            len = abs(kp_[11][0] - kp_[13][0])
            for t in range(len + 1):
                if line[start + t] == 0:
                    line[start + t] = 9
        if sorted_dist[i][0] == 'right_knee':
            start = min(kp_[12][0], kp_[14][0]) - X
            len = abs(kp_[12][0] - kp_[14][0])
            for t in range(len + 1):
                if line[start + t] == 0:
                    line[start + t] = 10
        if sorted_dist[i][0] == 'left_ankle':
            start = min(kp_[13][0], kp_[15][0]) - X
            len = abs(kp_[13][0] - kp_[15][0])
            for t in range(len + 1):
                if line[start + t] == 0:
                    line[start + t] = 11
        if sorted_dist[i][0] == 'right_ankle':
            start = min(kp_[14][0], kp_[16][0]) - X
            len = abs(kp_[14][0] - kp_[16][0])
            for t in range(len + 1):
                if line[start + t] == 0:
                    line[start + t] = 12
    return line

#对纵向的边框进行编码
def context_y(Y,line,kp_,sorted_dist,part_num):
    for i in range(part_num):
        if sorted_dist[i][0] == 'head':
            # head_h = max(2*abs(kp_[1][0]-kp_[2][0]),abs(kp_[3][0]-kp_[4][0]))
            head_h = abs(kp_[0][1] - (kp_[5][1]+kp_[6][1])/2)
            start = kp_[0][1]-int(head_h/2)-Y
            for t in range(head_h+1):
                if line[start+t] == 0:
                    line[start+t]=1
        if sorted_dist[i][0] == 'shoulder':
            start = min(kp_[6][1],kp_[7][1]) - Y
            len = abs(kp_[6][1] - kp_[7][1])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 2
        if sorted_dist[i][0] == 'left_elbow':
            start = min(kp_[5][1],kp_[7][1]) - Y
            len = abs(kp_[5][1]-kp_[7][1])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 3
        if sorted_dist[i][0] == 'right_elbow':
            start = min(kp_[6][1],kp_[8][1]) - Y
            len = abs(kp_[6][1]-kp_[8][1])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 4
        if sorted_dist[i][0] == 'left_arm':
            start = min(kp_[7][1],kp_[9][1]) - Y
            len = abs(kp_[7][1]-kp_[9][1])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 5
        if sorted_dist[i][0] == 'right_arm':
            start = min(kp_[8][1],kp_[10][1]) - Y
            len = abs(kp_[8][1]-kp_[10][1])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 6
        if sorted_dist[i][0] == 'hip':
            start = min(kp_[11][1],kp_[12][1]) - Y
            len = abs(kp_[11][1] - kp_[12][1])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 7
        #身体的部分比较复杂，四个点确定
        if sorted_dist[i][0] == 'body':
            Xs = (kp_[5]+kp_[6])/2
            Xh = (kp_[11]+kp_[12])/2
            len = abs(Xs[1]-Xh[1])
            start = min(Xs[1],Xh[1])
            for t in range(len+1):
                if line[start + t] == 0:
                    line[start + t] = 8
        if sorted_dist[i][0] == 'left_knee':
            start = min(kp_[11][1], kp_[13][1]) - Y
            len = abs(kp_[11][1] - kp_[13][1])
            for t in range(len + 1):
                if line[start + t] == 0:
                    line[start + t] = 9
        if sorted_dist[i][0] == 'right_knee':
            start = min(kp_[12][1], kp_[14][1]) - Y
            len = abs(kp_[12][1] - kp_[14][1])
            for t in range(len + 1):
                if line[start + t] == 0:
                    line[start + t] = 10
        if sorted_dist[i][0] == 'left_ankle':
            start = min(kp_[13][1], kp_[15][1]) - Y
            len = abs(kp_[13][1] - kp_[15][1])
            for t in range(len + 1):
                if line[start + t] == 0:
                    line[start + t] = 11
        if sorted_dist[i][0] == 'right_ankle':
            start = min(kp_[14][1], kp_[16][1]) - Y
            len = abs(kp_[14][1] - kp_[16][1])
            for t in range(len + 1):
                if line[start + t] == 0:
                    line[start + t] = 12
    return line

if __name__  == '__main__':
    part_points,part_mask = get_part_point(one_kp_points,one_kp_scores)
    # print('part_point',part_points)
    # print('length',len(part_points))
    dists=get_part_distance(part_mask,bboxes,part_points)
    bbox_code(bboxes,dists,one_kp_points)
    # print(dists[0])
    # print(dists[1])
    # print(dists[2])
    # print(dists[3])
    print(part_mask)


