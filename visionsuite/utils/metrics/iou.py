from shapely.geometry import Polygon
import numpy as np 

def handle_self_intersection(points):
    from shapely.geometry import (GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
                                mapping)
    from shapely.ops import polygonize, unary_union
    
    new_points = []
    line = LineString([[int(x), int(y)] for x, y in points + [points[0]]])

    polygons = list(polygonize(unary_union(line)))

    if len(polygons) > 1:
        print("The line is forming polygons by intersecting itself")
        for polygon in polygons:
            polygon = [list(item) for item in mapping(polygon)['coordinates'][0]]
            new_points.append(polygon[:-1])
    else:
        return [points]

    return new_points

def points2polygon(points):
    
    if isinstance(points, list):
        points = np.array(points)
        
    if isinstance(points, np.ndarray):
        points = Polygon(points)
    
    return points


def get_iou(points1, points2, filename=None):

    polygon1 = points2polygon(points1)
    polygon2 = points2polygon(points2)
    
    if not polygon1.is_valid:
        polygon1 = polygon1.buffer(0)  # 폴리곤의 유효성 수정

    if not polygon2.is_valid:
        polygon2 = polygon2.buffer(0)  # 폴리곤의 유효성 수정

    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    iou = intersection_area / union_area

    return iou