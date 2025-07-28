import numpy as np


def get_range(range_width, range_height, patch_width, patch_height):

    if range_width is None:
        range_width = int(patch_width / 5)
    if range_height is None:
        range_height = int(patch_height / 5)

    return range_width, range_height


def prevent_out_of_roi(roi, tl_x, tl_y, br_x, br_y, patch_width, patch_height):
    if tl_x < roi[0]:
        tl_x = roi[0]
        br_x = tl_x + patch_width
    if tl_y < roi[1]:
        tl_y = roi[1]
        br_y = tl_y + patch_height

    if br_x > roi[2]:
        br_x = roi[2]
        tl_x = br_x - patch_width
    if br_y > roi[3]:
        br_y = roi[3]
        tl_y = br_y - patch_height

    return tl_x, tl_y, br_x, br_y


def get_translated_roi(patch, roi, range_width=None, range_height=None, logger=None):
    """
    * patch: patch coords
    * roi: roi coords
    """

    if logger is not None:
        logger.assertion_log(
            patch[2] - patch[0] <= roi[2] - roi[0]
            and patch[3] - patch[1] <= roi[3] - roi[1],
            RuntimeError(f"Patch ({patch}) should be smaller than RoI ({roi})"),
            parent_fn=get_translated_roi.__name__,
        )

    tl_x, tl_y, br_x, br_y = patch[0], patch[1], patch[2], patch[3]
    patch_width = int(abs(br_x - tl_x))
    patch_height = int(abs(br_y - tl_y))

    xs, ys = [patch[0], patch[2]], [patch[1], patch[3]]
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    range_width, range_height = get_range(
        range_width, range_height, patch_width, patch_height
    )

    range_width_list = list(range(-range_width, 0)) + list(range(1, range_width + 1))
    range_height_list = list(range(-range_height, 0)) + list(range(1, range_height + 1))

    cx += np.random.choice(range_width_list)
    cy += np.random.choice(range_height_list)

    tl_x, tl_y, br_x, br_y = (
        int(cx - patch_width / 2),
        int(cy - patch_height / 2),
        int(cx + patch_width / 2),
        int(cy + patch_height / 2),
    )
    tl_x, tl_y, br_x, br_y = prevent_out_of_roi(
        roi, tl_x, tl_y, br_x, br_y, patch_width, patch_height
    )

    translated_patch = [tl_x, tl_y, br_x, br_y]
    if logger is not None:
        logger.assertion_log(
            translated_patch[2] - translated_patch[0] == patch_width
            and translated_patch[3] - translated_patch[1] == patch_height,
            RuntimeError(
                f"Translated patch has wrong width({translated_patch[2] - translated_patch[0]}) or height({translated_patch[3] - translated_patch[1]}), which must be width({patch_width}) or height({patch_height})"
            ),
            parent_fn=get_translated_roi.__name__,
        )
        logger.assertion_log(
            translated_patch[0] >= roi[0]
            and translated_patch[1] >= roi[1]
            and translated_patch[2] <= roi[2]
            and translated_patch[3] <= roi[3],
            RuntimeError(
                f"Translated patch has wrong width({translated_patch[2] - translated_patch[0]}) or height({translated_patch[3] - translated_patch[1]}), which must be in RoI({roi})"
            ),
            parent_fn=get_translated_roi.__name__,
        )

    return translated_patch


# def get_translated_roi(roi, img_width=None, img_height=None):
#     '''
#         * roi: list ([tl_x, tl_y, br_x, br_y])
#     '''
#     tl_x, tl_y, br_x, br_y = roi[0], roi[1], roi[2], roi[3]
#     roi_width = int(abs(br_x - tl_x))
#     roi_height = int(abs(br_y - tl_y))

#     xs, ys = [roi[0], roi[2]], [roi[1], roi[3]]
#     cx = int(np.mean(xs))
#     cy = int(np.mean(ys))

#     # FIXME: dist also should be random
#     dist_ratio = [2, 3, 4, 5]
#     move_x = int(roi_width/random.choice(dist_ratio)/2)
#     move_y = int(roi_height/random.choice(dist_ratio)/2)

#     centers = [[cx, cy], \
#                 [cx + move_x, cy], [cx - move_x, cy], \
#                 [cx, cy - move_y], [cx, cy + move_y], \
#                 [cx + move_x, cy + move_y], [cx + move_x, cy - move_y], \
#                 [cx - move_x, cy + move_y], [cx - move_x, cy - move_y]
#             ]

#     center_idx = random.choice(range(len(centers)))

#     cx = int(centers[center_idx][0])
#     cy = int(centers[center_idx][1])

#     br_offset_x = int(cx + roi_width/2 - br_x)
#     br_offset_y = int(cy + roi_height/2 - br_y)
#     if br_offset_x > 0:
#         cx -= br_offset_x
#     if br_offset_y > 0:
#         cy -= br_offset_y

#     tl_offset_x = int(cx - roi_width/2)
#     tl_offset_y = int(cy - roi_height/2)
#     if tl_offset_x < 0:
#         cx -= tl_offset_x
#     if tl_offset_y < 0:
#         cy -= tl_offset_y

#     new_roi = [int(cx - int(roi_width/2)), int(cy - int(roi_height/2)), \
#                                     int(cx + int(roi_width/2)), int(cy + int(roi_height/2))]

#     assert new_roi[2] - new_roi[0] == roi_width and new_roi[3] - new_roi[1] == roi_height, \
#             ValueError(f"New roi has wrong width({new_roi[2] - new_roi[0]}) or height({new_roi[3] - new_roi[1]}, which must be width({roi_width} or height({roi_height})")

#     return new_roi

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    roi = [0, 0, 60, 60]
    patch = [10, 10, 50, 50]

    translated_patch = get_translated_roi(patch, roi)

    patch_rect = [
        [patch[0], patch[1]],
        [patch[2], patch[1]],
        [patch[2], patch[3]],
        [patch[0], patch[3]],
        [patch[0], patch[1]],
    ]
    translated_patch_rect = [
        [translated_patch[0], translated_patch[1]],
        [translated_patch[2], translated_patch[1]],
        [translated_patch[2], translated_patch[3]],
        [translated_patch[0], translated_patch[3]],
        [translated_patch[0], translated_patch[1]],
    ]

    rects = [patch_rect, translated_patch_rect]
    cmap = get_cmap("tab10")
    num_colors = len(rects)
    color_list = [cmap(i) for i in range(num_colors)]
    for rect, color in zip(rects, color_list):
        xs, ys = [], []
        for point in rect:
            xs.append(point[0])
            ys.append(point[1])

        plt.plot(xs, ys, alpha=1, color=color)
    plt.savefig("./translate.png")
