import os
import pandas as pd


def read_coco_classes(filename):
    with open(filename) as f:
        lines = f.readlines()

    coco_dict_class = {}
    coco_dict_class_inverse = {}
    for l in lines:
        line_data = [x.strip() for x in l.split(',')]
        coco_dict_class[int(line_data[0])] = line_data[1]
        coco_dict_class_inverse[line_data[1]] = int(line_data[0])

    return coco_dict_class, coco_dict_class_inverse


def coco_ann_to_pd(coco, image_path):
    all_cats = coco.loadCats(coco.getCatIds())

    all_names = [cat['name'] for cat in all_cats]
    all_cocoids = [cat['id'] for cat in all_cats]
    all_super = [cat['supercategory'] for cat in all_cats]

    all_cocoids_to_names = {i: n for (n, i) in zip(all_names, all_cocoids)}
    all_cocoids_to_super = {i: n for (n, i) in zip(all_super, all_cocoids)}

    all_images_ids = list(coco.imgs.keys())

    img_ids, img_pathes = [], []
    img_heights, img_widths = [], []

    box_ids, box_classes, box_supers, areas = [], [], [], []
    box_xs, box_ys, box_ws, box_hs = [], [], [], []

    for img_id in all_images_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_ids)

        path_img = coco.loadImgs(img_id)[0]['file_name']
        path_img = os.path.join(image_path, path_img)

        for (i, a) in enumerate(ann):
            img_ids += [img_id]
            img_pathes += [path_img]
            img_heights += [coco.loadImgs(img_id)[0]['height']]
            img_widths += [coco.loadImgs(img_id)[0]['width']]

            box_ids += [a['category_id']]
            box_classes += [all_cocoids_to_names[a['category_id']]]
            box_supers += [all_cocoids_to_super[a['category_id']]]

            box_xs += [a['bbox'][0]]
            box_ys += [a['bbox'][1]]
            box_ws += [a['bbox'][2]]
            box_hs += [a['bbox'][3]]

            areas += [a['bbox'][2] * a['bbox'][3]]

    data = {'image_id': img_ids,
            'image_path': img_pathes,
            'image_width': img_widths,
            'image_height': img_heights,
            'box_id': box_ids,
            'box_class': box_classes,
            'box_super': box_supers,
            'box_x': box_xs,
            'box_y': box_ys,
            'box_w': box_ws,
            'box_h': box_hs,
            'area': areas,
            }

    df = pd.DataFrame.from_dict(data)

    return df


def coco_pred_to_pd(coco_res, image_path, coco_style=True):
    areas, iscrowds, class_names, category_id, scores, category_ids, ids = [], [], [], [], [], [], []

    if coco_style:
        coco_classes_names = os.path.join(os.getcwd(), os.path.dirname(__file__), "coco_classes.txt")
        coco_class_dict, _ = read_coco_classes(coco_classes_names)

    img_pathes = []

    img_ids = []
    box_xs, box_ys, box_ws, box_hs = [], [], [], []

    for k in coco_res.anns:
        ann_current = coco_res.anns[k]

        if coco_style:
            img_name = str(ann_current['image_id']).rjust(12, '0') + '.jpg'
            class_name = coco_class_dict[ann_current['category_id']]
        else:
            img_name = ann_current['image_name']
            class_name = ann_current['class_name']

        box_xs += [ann_current['bbox'][0]]
        box_ys += [ann_current['bbox'][1]]
        box_ws += [ann_current['bbox'][2]]
        box_hs += [ann_current['bbox'][3]]

        img_ids += [ann_current['image_id']]

        img_pathes += [os.path.join(image_path, img_name)]

        areas += [ann_current['area']]
        iscrowds += [ann_current['iscrowd']]
        class_names += [class_name]
        category_ids += [ann_current['category_id']]
        ids += [ann_current['id']]
        scores += [ann_current['score']]

    data = {'image_id': img_ids,
            'image_path': img_pathes,
            'box_class': class_names,
            'box_x': box_xs,
            'box_y': box_ys,
            'box_w': box_ws,
            'box_h': box_hs,
            'area': areas,
            'iscrowd': iscrowds,
            'category_id': category_ids,
            'id': ids,
            'score': scores
            }

    df = pd.DataFrame.from_dict(data)

    return df


def get_box_text_pred(df_pred_image):
    bboxes_pred = df_pred_image[['box_x', 'box_y', 'box_w', 'box_h']].values
    classes_names_pred = df_pred_image['box_class'].values
    score_pred = df_pred_image['score'].values
    bbox_text_pred = [classes_names_pred[i] + '[{:3.2f}]'.format(score_pred[i]) for i in range(len(score_pred))]

    return bboxes_pred, bbox_text_pred


def extract_info(df, df_pred_1, df_pred_2, condition_gt, condition_pred):

    # gt conditions
    conditions_image = [True] * df.shape[0]
    if len(condition_gt['classes']) > 0:
        conditions_image = conditions_image & df['box_class'].isin(condition_gt['classes'])
    if condition_gt['area_max'] != -1:
        conditions_image &= (df.area <= condition_gt['area_max'])
    if condition_gt['area_min'] != -1:
        conditions_image &= (df.area >= condition_gt['area_min'])

    df_selected = df.loc[conditions_image].reset_index(drop=True)
    df_selected_grouped = df_selected.groupby('image_id')

    # pred conditions (1)
    if len(df_pred_1) > 0:
        pred_conditions_1 = [True] * df_pred_1.shape[0]
        if len(condition_pred['classes']) > 0:
            pred_conditions_1 &= df_pred_1['box_class'].isin(condition_pred['classes'])
        if condition_pred['score_threshold'] != -1:
            pred_conditions_1 &= (df_pred_1['score'] > condition_pred['score_threshold'])

        df_pred_1_selected = df_pred_1.loc[pred_conditions_1].reset_index(drop=True)
        df_pred_1_selected_grouped = df_pred_1_selected.groupby('image_id')

    # pred conditions (2)
    if len(df_pred_2) > 0:
        pred_conditions_2 = [True] * df_pred_2.shape[0]
        if len(condition_pred['classes']) > 0:
            pred_conditions_2 &= df_pred_2['box_class'].isin(condition_pred['classes'])
        if condition_pred['score_threshold'] != -1:
            pred_conditions_2 &= (df_pred_2['score'] > condition_pred['score_threshold'])

        df_pred_2_selected = df_pred_2.loc[pred_conditions_2].reset_index(drop=True)
        df_pred_2_selected_grouped = df_pred_2_selected.groupby('image_id')

    # collect all info
    box_info = []
    for id_image, df_image in df_selected_grouped:

        # image path
        path_image = df_image['image_path'].iloc[0]  # pick the first

        # ground truth bboxes and text
        bboxes = df_image[['box_x', 'box_y', 'box_w', 'box_h']].values
        bbox_text = df_image['box_class'].values

        if len(df_pred_1) > 0 and id_image in df_pred_1_selected_grouped.groups.keys():
            df_pred_image = df_pred_1_selected_grouped.get_group(id_image)
            bboxes_pred, bbox_text_pred = get_box_text_pred(df_pred_image)
        else:
            bboxes_pred, bbox_text_pred = [], []

        if len(df_pred_2) > 0 and id_image in df_pred_2_selected_grouped.groups.keys():
            df_pred_image = df_pred_2_selected_grouped.get_group(id_image)
            bboxes_pred_2, bbox_text_pred_2 = get_box_text_pred(df_pred_image)
        else:
            bboxes_pred_2, bbox_text_pred_2 = [], []

        # collect all info
        box_info.append({'bboxes': bboxes,
                         'bbox_text': bbox_text,
                         'bboxes_pred': bboxes_pred,
                         'bbox_text_pred': bbox_text_pred,
                         'bboxes_pred_2': bboxes_pred_2,
                         'bbox_text_pred_2': bbox_text_pred_2,
                         'path_image': path_image
                         })

    return box_info
