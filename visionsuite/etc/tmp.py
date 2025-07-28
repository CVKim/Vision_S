from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# 두 polygon의 좌표 예시
polygon1_points = [[10, 10], [15, 13], [20, 10], [20, 20], [10, 20]]
polygon2_points = [[23, 15], [28, 15], [30, 30], [28, 20], [23, 20]]


def merge(points1, points2):
    # shapely Polygon 객체로 변환
    poly1 = Polygon(points1)
    poly2 = Polygon(points2)

    # 거리 기준
    merge_threshold = 5

    # 두 polygon 사이 거리
    distance = poly1.distance(poly2)
    print(f"거리: {distance:.2f}")

    if distance <= merge_threshold:
        # buffer를 줘서 확장 후 병합, 다시 buffer를 빼서 원래로 복원
        buffered_union = unary_union([poly1.buffer(merge_threshold), poly2.buffer(merge_threshold)])
        merged = buffered_union.buffer(-merge_threshold)
        print("병합 완료 (거리 기준).")
    else:
        merged = [poly1, poly2]
        print("병합 조건 미달.")

    # 시각화 함수
    def plot_polygon(p, color='blue', label=None):
        if isinstance(p, Polygon):
            x, y = p.exterior.xy
            plt.plot(x, y, color=color, label=label)
        else:  # MultiPolygon
            for geom in p.geoms:
                x, y = geom.exterior.xy
                plt.plot(x, y, color=color, label=label)

    # 시각화
    plt.figure(figsize=(6, 6))
    plot_polygon(poly1, 'blue', 'Polygon 1')
    plot_polygon(poly2, 'green', 'Polygon 2')
    plot_polygon(merged, 'red', 'Merged')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.title("Polygon Merge by Distance ≤ 5")
    plt.grid(True)
    plt.show()
