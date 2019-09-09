import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def plot_boxes(ax, bboxes, bboxes_text, color_bbox, color_text,
               text_pos='left_top', print_bboxes=False):

    for i_box in range(len(bboxes)):
        if print_bboxes:
            print("box [#{}]: (x, y, w, h) "
                  "({:.2f}, {:.2f}, {:.2f}, {:.2f}) (class): {}".format(i_box, bboxes[i_box][0],
                                                                        bboxes[i_box][1],
                                                                        bboxes[i_box][2],
                                                                        bboxes[i_box][3],
                                                                        bboxes_text[i_box]))

        ax.add_patch(
            Rectangle((bboxes[i_box][0], bboxes[i_box][1]),  # (x,y)
                      bboxes[i_box][2],  # width
                      bboxes[i_box][3],  # height
                      alpha=1,
                      facecolor='none',
                      edgecolor=color_bbox,  # color
                      linewidth=1))

        # position of the text box
        if text_pos == 'right_top':
            (x, y) = bboxes[i_box][0] + bboxes[i_box][2], bboxes[i_box][1]
        elif text_pos == 'right_bottom':
            (x, y) = bboxes[i_box][0] + bboxes[i_box][2], bboxes[i_box][1] + bboxes[i_box][3]
        else:
            (x, y) = bboxes[i_box][0], bboxes[i_box][1]

        ax.text(x, y,
                bboxes_text[i_box],  # text itself
                color=color_text[1],  # text color
                bbox=dict(facecolor=color_text[0],
                          edgecolor=color_text[2], boxstyle='round'))


def plot_single_image(current_image_info, fig_size=12, make_print=False):

    # gt appereance
    color_bbox = 'red'
    color_text = ('red', 'black', 'white')

    # pred appereance
    color_bbox_pred = 'green'
    color_text_pred = ('green', 'black', 'white')

    # pred (2) appereance
    color_bbox_pred_2 = 'blue'
    color_text_pred_2 = ('blue', 'white', 'white')

    # load image
    img = Image.open(current_image_info['path_image']).convert('RGB')

    _ = plt.figure(figsize=(fig_size, fig_size))
    ax = plt.subplot()
    ax.set_aspect('equal')
    plt.imshow(img)

    # ground truth
    if make_print:
        print("= ground truth:")
    plot_boxes(ax, current_image_info['bboxes'],
               current_image_info['bbox_text'],
               color_bbox,
               color_text,
               text_pos='left_top',
               print_bboxes=make_print)

    # predictions
    if make_print:
        print("= prediction truth:")
    plot_boxes(ax,
               current_image_info['bboxes_pred'],
               current_image_info['bbox_text_pred'],
               color_bbox_pred,
               color_text_pred,
               text_pos='right_top',
               print_bboxes=make_print)

    # predictions
    if make_print:
        print("= prediction truth (2):")
    plot_boxes(ax,
               current_image_info['bboxes_pred_2'],
               current_image_info['bbox_text_pred_2'],
               color_bbox_pred_2,
               color_text_pred_2,
               text_pos='right_bottom',
               print_bboxes=make_print)
