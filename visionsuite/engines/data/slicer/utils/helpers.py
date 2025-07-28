import matplotlib.pyplot as plt
import shapely
from matplotlib.cm import get_cmap


def vis_slicer():
    pass


def plot_poinsts_list(points_list, output_filename):

    cmap = get_cmap("tab10")
    num_colors = len(points_list)
    color_list = [cmap(i) for i in range(num_colors)]

    fig, ax = plt.subplots()
    for points, color in zip(points_list, color_list):
        xs, ys = [], []
        for point in points:
            xs.append(point[0])
            ys.append(point[1])
        plt.plot(xs, ys, alpha=1, color=color)

    ax.set_aspect("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polygon Intersection")
    plt.grid(True)
    plt.savefig(output_filename)


def plot_polygons(polygon_list):

    cmap = get_cmap("tab10")
    num_colors = len(polygon_list)
    color_list = [cmap(i) for i in range(num_colors)]

    fig, ax = plt.subplots()
    for polygon, color in zip(polygon_list, color_list):
        if isinstance(polygon, shapely.geometry.polygon.Polygon):
            if polygon.geom_type == "Polygon":
                x, y = polygon.exterior.xy
                plt.plot(x, y, alpha=1, color=color)
            elif polygon.geom_type == "MultiPolygon":
                for part in polygon:
                    x, y = part.exterior.xy
                    plt.plot(x, y, alpha=1, color=color)

    ax.set_aspect("equal")
    plt.xlim(-10, 522)
    plt.ylim(1466, 2020)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polygon Intersection")
    plt.grid(True)
    plt.savefig("/HDD/test.png")


if __name__ == "__main__":
    from athena.src.data.polygons import get_intersected_points

    points1 = [[3, 5], [6, 2], [6, 5], [3, 2], [3, 5]]
    points2 = [[0, 7], [8, 7], [8, 0], [0, 0], [0, 7]]

    # points1 = reorder_points(points1)

    output_filename = "/HDD/case1.png"
    plot_poinsts_list([points1] + [points2], output_filename)

    inter = get_intersected_points(points1, points2, False, True)
    print(inter)
    output_filename = "/HDD/case2.png"
    plot_poinsts_list([points2] + inter, output_filename)

    # roi = [4522, 2104, 5546, 3128]
    # points1 = close_roi(roi)
    # print(points1)
    # points2 = [        [
    #       4211.0,
    #       2390.0
    #     ],
    #     [
    #       4165.0,
    #       2414.0
    #     ],
    #     [
    #       4132.0,
    #       2441.0
    #     ],
    #     [
    #       4095.0,
    #       2481.0
    #     ],
    #     [
    #       4082.0,
    #       2496.0
    #     ],
    #     [
    #       4107.0,
    #       2536.0
    #     ],
    #     [
    #       4107.0,
    #       2536.0
    #     ],
    #     [
    #       4162.0,
    #       2514.0
    #     ],
    #     [
    #       4185.0,
    #       2530.0
    #     ],
    #     [
    #       4184.0,
    #       2550.0
    #     ],
    #     [
    #       4212.0,
    #       2545.0
    #     ],
    #     [
    #       4208.0,
    #       2604.0
    #     ],
    #     [
    #       4232.0,
    #       2649.0
    #     ],
    #     [
    #       4229.0,
    #       2674.0
    #     ],
    #     [
    #       4153.0,
    #       2685.0
    #     ],
    #     [
    #       4131.0,
    #       2694.0
    #     ],
    #     [
    #       4073.0,
    #       2777.0
    #     ],
    #     [
    #       4039.0,
    #       2783.0
    #     ],
    #     [
    #       4005.0,
    #       2811.0
    #     ],
    #     [
    #       4014.0,
    #       2857.0
    #     ],
    #     [
    #       4069.0,
    #       2833.0
    #     ],
    #     [
    #       4492.0,
    #       2717.0
    #     ],
    #     [
    #       4776.0,
    #       2694.0
    #     ],
    #     [
    #       5011.0,
    #       2712.0
    #     ],
    #     [
    #       5093.0,
    #       2722.0
    #     ],
    #     [
    #       5133.0,
    #       2685.0
    #     ],
    #     [
    #       5142.0,
    #       2649.0
    #     ],
    #     [
    #       5118.0,
    #       2609.0
    #     ],
    #     [
    #       5088.0,
    #       2604.0
    #     ],
    #     [
    #       5065.0,
    #       2623.0
    #     ],
    #     [
    #       5040.0,
    #       2637.0
    #     ],
    #     [
    #       5001.0,
    #       2636.0
    #     ],
    #     [
    #       4976.0,
    #       2634.0
    #     ],
    #     [
    #       4955.0,
    #       2614.0
    #     ],
    #     [
    #       4938.0,
    #       2591.0
    #     ],
    #     [
    #       4950.0,
    #       2566.0
    #     ],
    #     [
    #       4985.0,
    #       2542.0
    #     ],
    #     [
    #       5012.0,
    #       2528.0
    #     ],
    #     [
    #       5033.0,
    #       2539.0
    #     ],
    #     [
    #       5012.0,
    #       2555.0
    #     ],
    #     [
    #       4972.0,
    #       2575.0
    #     ],
    #     [
    #       4958.0,
    #       2596.0
    #     ],
    #     [
    #       4985.0,
    #       2606.0
    #     ],
    #     [
    #       5027.0,
    #       2582.0
    #     ],
    #     [
    #       5050.0,
    #       2564.0
    #     ],
    #     [
    #       5060.0,
    #       2541.0
    #     ],
    #     [
    #       5103.0,
    #       2548.0
    #     ],
    #     [
    #       5123.0,
    #       2534.0
    #     ],
    #     [
    #       5143.0,
    #       2540.0
    #     ],
    #     [
    #       5142.0,
    #       2508.0
    #     ],
    #     [
    #       5165.0,
    #       2492.0
    #     ],
    #     [
    #       5177.0,
    #       2521.0
    #     ],
    #     [
    #       5166.0,
    #       2554.0
    #     ],
    #     [
    #       5186.0,
    #       2570.0
    #     ],
    #     [
    #       5217.0,
    #       2564.0
    #     ],
    #     [
    #       5253.0,
    #       2552.0
    #     ],
    #     [
    #       5278.0,
    #       2560.0
    #     ],
    #     [
    #       5283.0,
    #       2584.0
    #     ],
    #     [
    #       5271.0,
    #       2615.0
    #     ],
    #     [
    #       5248.0,
    #       2625.0
    #     ],
    #     [
    #       5233.0,
    #       2636.0
    #     ],
    #     [
    #       5240.0,
    #       2655.0
    #     ],
    #     [
    #       5257.0,
    #       2656.0
    #     ],
    #     [
    #       5274.0,
    #       2674.0
    #     ],
    #     [
    #       5260.0,
    #       2688.0
    #     ],
    #     [
    #       5242.0,
    #       2686.0
    #     ],
    #     [
    #       5221.0,
    #       2689.0
    #     ],
    #     [
    #       5209.0,
    #       2710.0
    #     ],
    #     [
    #       5166.0,
    #       2737.0
    #     ],
    #     [
    #       5349.0,
    #       2797.0
    #     ],
    #     [
    #       5574.0,
    #       2883.0
    #     ],
    #     [
    #       5804.0,
    #       3012.0
    #     ],
    #     [
    #       5842.0,
    #       3031.0
    #     ],
    #     [
    #       5826.0,
    #       2985.0
    #     ],
    #     [
    #       5811.0,
    #       2963.0
    #     ],
    #     [
    #       5810.0,
    #       2921.0
    #     ],
    #     [
    #       5842.0,
    #       2914.0
    #     ],
    #     [
    #       5868.0,
    #       2907.0
    #     ],
    #     [
    #       5894.0,
    #       2842.0
    #     ],
    #     [
    #       5907.0,
    #       2800.0
    #     ],
    #     [
    #       5897.0,
    #       2765.0
    #     ],
    #     [
    #       5851.0,
    #       2732.0
    #     ],
    #     [
    #       5789.0,
    #       2697.0
    #     ],
    #     [
    #       5723.0,
    #       2657.0
    #     ],
    #     [
    #       5665.0,
    #       2635.0
    #     ],
    #     [
    #       5627.0,
    #       2641.0
    #     ],
    #     [
    #       5604.0,
    #       2642.0
    #     ],
    #     [
    #       5577.0,
    #       2612.0
    #     ],
    #     [
    #       5554.0,
    #       2595.0
    #     ],
    #     [
    #       5543.0,
    #       2574.0
    #     ],
    #     [
    #       5510.0,
    #       2585.0
    #     ],
    #     [
    #       5490.0,
    #       2558.0
    #     ],
    #     [
    #       5466.0,
    #       2541.0
    #     ],
    #     [
    #       5425.0,
    #       2531.0
    #     ],
    #     [
    #       5388.0,
    #       2532.0
    #     ],
    #     [
    #       5358.0,
    #       2508.0
    #     ],
    #     [
    #       5366.0,
    #       2468.0
    #     ],
    #     [
    #       5326.0,
    #       2424.0
    #     ],
    #     [
    #       5266.0,
    #       2381.0
    #     ],
    #     [
    #       5221.0,
    #       2370.0
    #     ],
    #     [
    #       5181.0,
    #       2381.0
    #     ],
    #     [
    #       5106.0,
    #       2388.0
    #     ],
    #     [
    #       5075.0,
    #       2408.0
    #     ],
    #     [
    #       5042.0,
    #       2415.0
    #     ],
    #     [
    #       4991.0,
    #       2415.0
    #     ],
    #     [
    #       4684.0,
    #       2423.0
    #     ],
    #     [
    #       4407.0,
    #       2434.0
    #     ],
    #     [
    #       4334.0,
    #       2439.0
    #     ],
    #     [
    #       4263.0,
    #       2409.0
    #     ]
    # ]

    # output_filename = '/HDD/case1.png'
    # plot_poinsts_list([points1] + [points2], output_filename)

    # inter = get_intersected_points(points1, points2, False, True)
    # print(inter)
    # output_filename = '/HDD/case2.png'
    # plot_poinsts_list([points2] + inter, output_filename)
