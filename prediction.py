import pandas as pd
from catboost import CatBoostRegressor
import math
import os
from pathlib import Path
from mmdet.apis import inference_detector

HOME = Path.home()

#import logging
#logging.basicConfig(
#    filename=os.path.join(HOME,'prediction/codes/logs/dummy.log'),
#    format='%(asctime)s:%(levelname)s:%(message)s',
#    datefmt='%Y/%m/%d/ %I:%M:%S %p',
#    level=logging.WARNING
#)

import warnings
warnings.filterwarnings('ignore')

MAX_HEIGHT = 1080
MAX_WIDTH = 810
DISTANCE_SIZE = 100

def get_stat():
    return {'waist_avg': {2: 41.725,
              3: 42.33841463414634,
              4: 44.68813559322034,
              5: 46.49684684684684,
              6: 48.00972839506173},
             'waist_std': {2: 1.8533139850745,
              3: 2.2742967815673025,
              4: 2.4666465519458938,
              5: 2.2995407573062705,
              6: 2.5376198198919866},
             'thigh_avg': {2: 23.60909090909091,
              3: 26.129925925925924,
              4: 28.359250000000003,
              5: 29.96778181818182,
              6: 31.27675581395349},
             'thigh_std': {2: 1.461817051107659,
              3: 2.266453825791393,
              4: 2.530842608869538,
              5: 2.635720848710451,
              6: 2.60089950926958},
             'head_avg': {2: 38.275,
              3: 42.71707317073171,
              4: 45.00257894736842,
              5: 47.078101851851855,
              6: 48.55458333333333},
             'head_std': {2: 2.991844976538119,
              3: 1.9417635409341,
              4: 2.0004810876905923,
              5: 3.05070152171201,
              6: 1.9205859686177569},
             'weight_avg': {2: 5.7,
              3: 7.945121951219512,
              4: 9.823529411764708,
              5: 11.789473684210526,
              6: 14.198837209302324},
             'weight_std': {2: 0.6536610186271835,
              3: 1.1728568518028264,
              4: 1.290624988684069,
              5: 1.593313929538813,
              6: 4.139295029095328},
             }


def get_regressor():
    m_thigh = CatBoostRegressor(iterations=1000,random_seed=0,learning_rate=0.05,depth=10,min_data_in_leaf=5,border_count=64,l2_leaf_reg=6,loss_function='MAE',eval_metric='MAE')
    m_head = CatBoostRegressor(iterations=1000,random_seed=0,learning_rate=0.05,depth=10,min_data_in_leaf=5,border_count=64,l2_leaf_reg=6,loss_function='MAE',eval_metric='MAE')
    m_weight = CatBoostRegressor(iterations=1000,random_seed=0,learning_rate=0.05,depth=10,min_data_in_leaf=5,border_count=64,l2_leaf_reg=6,loss_function='MAE',eval_metric='MAE')
    m_waist = CatBoostRegressor(iterations=2000,random_seed=0,learning_rate=0.01,depth = 10,min_data_in_leaf=7,border_count=64,l2_leaf_reg = 6,loss_function='MAE',eval_metric='MAE')
    return m_waist, m_thigh, m_head, m_weight


def zptile(x, m, s):
    z_score = (x-m)/s
    return round(1-  .5 * (math.erf(z_score / 2 ** .5) + 1), 3)


def check_predict_outlier(stage, waist, thigh, head, weight, stage_dict):
    stage_dict = stage_dict.copy()
    min_m = 0.3
    max_m = 3
    m = stage_dict['waist_avg'][stage]
    m1 = m * min_m
    m2 = m * max_m
    if (waist < m1) | (waist > m2):
        return False

    m = stage_dict['thigh_avg'][stage]
    m1 = m * min_m
    m2 = m * max_m
    if (thigh < m1) | (thigh > m2):
        return False

    m = stage_dict['head_avg'][stage]
    m1 = m * min_m
    m2 = m * max_m
    if (head < m1) | (head > m2):
        return False

    m = stage_dict['weight_avg'][stage]
    m1 = m * min_m
    m2 = m * max_m
    if(weight < m1) | (weight > m2):
        return False

    return True


def compute_dist(xt, yt, xb, yb):
    return xt ** 2 + yt ** 2, xb ** 2 + yb ** 2


def get_line_distances(front_line, back_line, area):
    distances = []
    front_ratio = int(len(front_line) / (DISTANCE_SIZE - 1))
    back_ratio = int(len(back_line) / (DISTANCE_SIZE - 1))

    for i in range(DISTANCE_SIZE):
        # get ratio index
        front_index = front_ratio * i
        back_index = back_ratio * i

        # for index number
        if (front_ratio * i) == len(front_line):
            front_index -= 1
        if (back_ratio * i) == len(back_line):
            back_index -= 1

        # get points
        x1, y1 = front_line[front_index]
        x2, y2 = back_line[back_index]

        # get distance
        a = x1 - x2
        b = y1 - y2
        pixel_distance = math.sqrt(a ** 2 + b ** 2)
        distance = pixel_distance / area
        distances.append(distance)

    return distances


def get_front_back_line(mask, bound_box):
    x1, y1, x2, y2 = bound_box
    x12 = x2 - x1
    y12 = y2 - y1
    diff = int((y12 - x12))

    left_line = []
    min_dist_top, min_dist_bottom = compute_dist(MAX_WIDTH + diff, MAX_HEIGHT, MAX_WIDTH + diff, MAX_HEIGHT)
    for h in range(y1 - 1, min(MAX_HEIGHT - 1, y2 + 1), 1):
        for w in range(x1 - 1, min(MAX_WIDTH - 1, x2 + 1), 1):
            if mask[h][w] == 255:
                left_line.append([w, h])
                dist_top, dist_bottom = compute_dist(w + diff, h - y1, w + diff, y2 - h)
                if dist_top < min_dist_top:
                    min_dist_top = dist_top
                    min_top_left_w, min_top_left_h = w, h
                if dist_bottom < min_dist_bottom:
                    min_dist_bottom = dist_bottom
                    min_bottom_left_w, min_bottom_left_h = w, h
                break

    right_line = []
    min_dist_top, min_dist_bottom = compute_dist(MAX_WIDTH + diff, MAX_HEIGHT, MAX_WIDTH + diff, MAX_HEIGHT)
    for h in range(y1 - 1, min(MAX_HEIGHT - 1, y2 + 1), 1):
        for w in range(min(MAX_WIDTH - 1, x2 + 1), x1 - 1, -1):
            if mask[h][w] == 255:
                right_line.append([w, h])
                dist_top, dist_bottom = compute_dist(x2 + diff - w, h - y1, x2 + diff - w, y2 - h)
                if dist_top < min_dist_top:
                    min_dist_top = dist_top
                    min_top_right_w, min_top_right_h = w, h
                if dist_bottom < min_dist_bottom:
                    min_dist_bottom = dist_bottom
                    min_bottom_right_w, min_bottom_right_h = w, h
                break

    up_line = []
    for w in range(x1 - 1, min(MAX_WIDTH - 1, x2 + 1), 1):
        for h in range(y1 - 1, min(MAX_HEIGHT - 1, y2 + 1), 1):
            if mask[h][w] == 255:
                up_line.append([w, h])
                break

    down_line = []
    for w in range(x1 - 1, min(MAX_WIDTH - 1, x2 + 1), 1):
        for h in range(min(MAX_HEIGHT - 1, y2 + 1), 0, -1):
            if mask[h][w] == 255:
                down_line.append([w, h])
                break

    if len(left_line) == 0 or len(right_line) == 0 or len(up_line) == 0 or len(down_line) == 0:
        #logging.error('No found detected Line. Check Mask left line length{}, right line length{}, up line length{}, downline length{}'.format(
        #        len(left_line), len(right_line), len(up_line), len(down_line)))
        return [], [], [], [], 0

    # get corners
    x12 = x2 - x1
    y12 = y2 - y1
    diff = int((y12 - x12))

    con = [x for x in down_line if x in left_line]
    max_score = math.pow((con[0][0] + diff), 2) + math.pow((con[0][1] - y2), 2)
    left_bottom = [con[0][0], con[0][1]]
    for x in con:
        score = math.pow((x[0] + diff), 2) + math.pow((y2 - x[1]), 2)
        if score < max_score:
            max_score = score
            left_bottom[0] = x[0]
            left_bottom[1] = x[1]

    con = [x for x in up_line if x in left_line]
    max_score = math.pow((con[0][0] + diff), 2) + math.pow((con[0][1] - y1), 2)
    left_top = [con[0][0], con[0][1]]
    for x in con:
        score = math.pow((x[0] + diff), 2) + math.pow((x[1] - y1), 2)
        if score < max_score:
            max_score = score
            left_top[0] = x[0]
            left_top[1] = x[1]

    con = [x for x in down_line if x in right_line]
    max_score = math.pow((x2 + diff - con[0][0]), 2) + math.pow((y2 - con[0][1]), 2)
    right_bottom = [con[0][0], con[0][1]]
    for x in con:
        score = math.pow((x2 + diff - x[0]), 2) + math.pow((y2 - x[1]), 2)
        if score < max_score:
            max_score = score
            right_bottom[0] = x[0]
            right_bottom[1] = x[1]

    con = [x for x in up_line if x in right_line]
    max_score = math.pow((x2 + diff - con[0][0]), 2) + math.pow((con[0][1] - y1), 2)
    right_top = [con[0][0], con[0][1]]
    for x in con:
        score = math.pow((x2 + diff - x[0]), 2) + math.pow((x[1] - y1), 2)
        if score < max_score:
            max_score = score
            right_top[0] = x[0]
            right_top[1] = x[1]

    # get lines
    left_line = [x for x in left_line if x[1] >= left_top[1] and x[1] <= left_bottom[1]]
    right_line = [x for x in right_line if x[1] >= right_top[1] and x[1] <= right_bottom[1]]
    top_line = [x for x in up_line if x[0] >= left_top[0] and x[0] <= right_top[0]]
    bottom_line = [x for x in down_line if x[0] >= left_bottom[0] and x[0] <= right_bottom[0]]

    if len(left_line) == 0 or len(right_line) == 0 or len(top_line) == 0 or len(bottom_line) == 0:
        return [], [], [], [], 0
    if len(left_line) > len(right_line):    
        return right_line, left_line, top_line, bottom_line, len(right_line)
    else:        
        return left_line, right_line, top_line, bottom_line, len(left_line)


def get_a_dist_list_w_inference(img, mm_model):
    if (img.shape[0] != MAX_HEIGHT) | (img.shape[1] != MAX_WIDTH):    
        # logging.error(f'pixel 에러 height:{img.shape[0]} width:{img.shape[1]}')
        dist = []
    else:
        result = inference_detector(mm_model, img)
        mask = result[1][0][0].copy()
        mask = mask * 1
        mask[mask > 0] = 255
        bbox = list(map(int, result[0][0][0][:4]))
        front_line, back_line, top_line, bottom_line, pixel_standard = get_front_back_line(mask, bbox)
        dist = get_line_distances(front_line, back_line, pixel_standard)
    return dist


def predict(mm_model, jpg_array, stage):
    try: 
        dist_arr = get_a_dist_list_w_inference(jpg_array, mm_model)
        if len(dist_arr) > 0:            
            df = pd.DataFrame([dist_arr])

            m_waist, m_thigh, m_head, m_weight = get_regressor()
            
            m_weight.load_model(os.path.join(HOME,f'prediction/models/regression/weight/weight_{stage}_SelN_MeanY.pkl'))
            m_head.load_model(os.path.join(HOME,f'prediction/models/regression/head/head_{stage}_SelN_MeanY.pkl'))
            m_thigh.load_model(os.path.join(HOME,f'prediction/models/regression/thigh/thigh_{stage}_SelN_MeanY.pkl'))
            m_waist.load_model(os.path.join(HOME,f'prediction/models/regression/waist/waist_{stage}.pkl'))

            if (stage==2)|(stage==3)|(stage==5):
                waist = round(m_waist.predict(df)[0], 1)
            else:
                sel_fea = [0, 1, 2, 9, 13, 14, 21, 22, 25, 26, 28, 35, 49, 50, 53, 54, 55,
                           56, 62, 63, 65, 66, 67, 68, 72, 73, 75, 76, 77, 81, 82, 84, 85, 86,
                           90, 93, 94, 96, 99]  # need to copy "selected features" manually colab 7_치수학습_infer_mae_t2023_val2023_waist_final0610.ipynb
                waist = round(m_waist.predict(df[sel_fea])[0], 1)
            
            # waist 이외는 mean 사용
            a_mean = df.mean(axis=1)
            a_sum = df.sum(axis=1)
            df['mean'] = a_mean
            df['sum'] = a_sum
            thigh = round(m_thigh.predict(df)[0], 1)
            head = round(m_head.predict(df)[0], 1)
            weight = round(m_weight.predict(df)[0], 1)

            stage_dict = get_stat()        
            if check_predict_outlier(stage, waist, thigh, head, weight, stage_dict):
                pass
            else:                
                #logging.error('abnormal predicted size', stage, waist, thigh, head, weight)
                return -999, -999, -999, -999, -999, -999, -999, -999
            
            m = stage_dict['waist_avg'][stage]
            s = stage_dict['waist_std'][stage]
            z_waist = zptile(waist, m, s)

            m = stage_dict['thigh_avg'][stage]
            s = stage_dict['thigh_std'][stage]
            z_thigh = zptile(thigh, m, s)

            m = stage_dict['head_avg'][stage]
            s = stage_dict['head_std'][stage]
            z_head = zptile(head, m, s)

            m = stage_dict['weight_avg'][stage]
            s = stage_dict['weight_std'][stage]
            z_weight = zptile(weight, m, s)

            return waist, thigh, head, weight, z_waist, z_thigh, z_head, z_weight
        else:
            #logging.error('dist_arr=[]')
            return -999, -999, -999, -999, -999, -999, -999, -999
    except Exception as e:
        #logging.error(e)
        return -999, -999, -999, -999, -999, -999, -999, -999