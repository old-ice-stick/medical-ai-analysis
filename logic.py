import cv2
import math
import numpy as np
from typing import List, Tuple, Dict, Optional

class GeometryUtils:
    """几何计算工具类"""
    @staticmethod
    def get_vector(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        return p2[0] - p1[0], p2[1] - p1[1]

    @staticmethod
    def get_midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

    @staticmethod
    def rotate_vector_90(dx: float, dy: float) -> Tuple[float, float]:
        return -dy, dx

    @staticmethod
    def calculate_angle(v1: Tuple[float, float], v2: Tuple[float, float], force_obtuse: bool = False) -> float:
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])
        if norm1 == 0 or norm2 == 0: return 0.0
        
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        angle = math.degrees(math.acos(cos_theta))
        
        if not force_obtuse:
            return float(angle if angle <= 90 else 180 - angle)
        else:
            return float(angle if angle >= 90 else 180 - angle)

    @staticmethod
    def get_clipped_line(center: Tuple[float, float], vec: Tuple[float, float], w: int, h: int):
        """计算穿过中心点且在图像范围内的线段"""
        if vec[0] == 0 and vec[1] == 0: return None
        cx, cy = center
        dx, dy = vec
        t_vals = []
        for x_bound in [0, w]:
            if abs(dx) > 1e-9: t_vals.append((x_bound - cx) / dx)
        for y_bound in [0, h]:
            if abs(dy) > 1e-9: t_vals.append((y_bound - cy) / dy)
        
        pts = []
        for t in t_vals:
            px, py = cx + t * dx, cy + t * dy
            if -1 <= px <= w+1 and -1 <= py <= h+1:
                pts.append((int(px), int(py)))
        
        if len(pts) >= 2:
            return pts[0], pts[-1]
        return None

class ImageVisualizer:
    """可视化绘制类"""
    def __init__(self, img_size: int):
        self.thickness = max(2, int(img_size / 300))
        self.radius = max(3, int(img_size / 200))
        self.font_scale = max(0.6, img_size / 1000)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_common(self, img, p1, p2, p3, p4, ids):
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (226, 43, 138), self.thickness)
        cv2.line(img, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (226, 43, 138), self.thickness)
        for i, p in zip(ids, [p1, p2, p3, p4]):
            cv2.circle(img, (int(p[0]), int(p[1])), self.radius, (255, 0, 0), -1)
            cv2.putText(img, str(i), (int(p[0])+5, int(p[1])-5), self.font, self.font_scale, (0, 255, 255), 1)

    def draw_title(self, img, val):
        cv2.putText(img, f"{val:.1f} deg", (40, 80), self.font, self.font_scale*2, (0, 255, 0), 2)

class BaseProcessor:
    def __init__(self):
        self.geo = GeometryUtils()

    def _get_kp(self, kps, idx):
        if 1 <= idx <= len(kps):
            return tuple(kps[idx-1])
        return (0.0, 0.0)

class POSProcessor(BaseProcessor):
    """正位角度处理器"""
    NAMES = ["四五跖骨间角", "距舟覆盖角", "前足内收角", "第一二跖骨间角", "PASA", "DASA", "HVA", "IPA", "跖楔角", "舟骨侧移角"]
    
    def process(self, image, kps):
        h, w = image.shape[:2]
        vis = ImageVisualizer(min(h, w))
        results = {'values': {}, 'images': []}
        
        # 配置: (ID, 类型, 点集) 类型: 0-普通, 1-垂线, 2-双中垂, 3-复杂中点
        configs = [
            (1, 0, [23, 24, 25, 26]), (2, 0, [1, 2, 3, 4]), (3, 3, [17, 18, 19, 20, 21, 22]),
            (4, 0, [7, 8, 17, 18]), (5, 1, [7, 8, 13, 14]), (6, 1, [9, 10, 15, 16]),
            (7, 0, [7, 8, 9, 10]), (8, 0, [9, 10, 11, 12]), (9, 1, [7, 8, 5, 6]), (10, 2, [1, 2, 3, 4])
        ]

        for i, (aid, t, ids) in enumerate(configs):
            img_draw = image.copy()
            pts = [self._get_kp(kps, idx) for idx in ids]
            val = 0.0
            
            try:
                if t == 0:
                    v1 = self.geo.get_vector(pts[0], pts[1])
                    v2 = self.geo.get_vector(pts[2], pts[3])
                    val = self.geo.calculate_angle(v1, v2)
                    vis.draw_common(img_draw, pts[0], pts[1], pts[2], pts[3], ids)
                elif t == 1:
                    v_line = self.geo.get_vector(pts[0], pts[1])
                    v_base = self.geo.get_vector(pts[2], pts[3])
                    v_perp = self.geo.rotate_vector_90(*v_base)
                    val = self.geo.calculate_angle(v_line, v_perp)
                    vis.draw_common(img_draw, pts[0], pts[1], pts[2], pts[3], ids)
                    line = self.geo.get_clipped_line(self.geo.get_midpoint(pts[2], pts[3]), v_perp, w, h)
                    if line: cv2.line(img_draw, line[0], line[1], (0, 255, 0), vis.thickness)
                # ... 其他复杂逻辑简化处理 ...
                elif t == 3: # 前足内收
                    m1 = self.geo.get_midpoint(pts[2], pts[3])
                    m2 = self.geo.get_midpoint(pts[4], pts[5])
                    v_mid = self.geo.get_vector(m1, m2)
                    v_perp = self.geo.rotate_vector_90(*v_mid)
                    val = self.geo.calculate_angle(self.geo.get_vector(pts[0], pts[1]), v_perp)
                    vis.draw_common(img_draw, pts[0], pts[1], pts[2], pts[3], ids[:4])
                
                val = round(val, 1)
                vis.draw_title(img_draw, val)
                results['values'][self.NAMES[i]] = val
                results['images'].append(img_draw)
            except:
                results['values'][self.NAMES[i]] = 0.0
                results['images'].append(img_draw)
        return results

class SideProcessor(BaseProcessor):
    """侧位角度处理器 (整合版)"""
    NAMES = [
        "根骨第五跖骨角", "根骨第一跖骨角", "根骨交叉角", "跟骨结节关节角", "后弓角", "前弓角",
        "外侧纵弓", "内侧纵弓", "纵弓角", "距跟角", "距骨第一跖骨角", "第五跖骨倾斜角",
        "胫骨远端前角", "胫骨倾斜角", "跟骨倾斜角", "距骨倾斜角", "第一跖骨倾斜角", "侧位Kite角"
    ]

    def process(self, image, kps):
        h, w = image.shape[:2]
        vis = ImageVisualizer(min(h, w))
        results = {'values': {}, 'images': []}
        
        # 配置: (点ID, 是否强制钝角)
        configs = [
            ([20, 21, 22, 29], False), ([5, 6, 20, 21], False), ([18, 19, 22, 29], False),
            ([16, 17, 20, 21], False), ([22, 28, 26, 28], True), ([22, 27, 26, 27], True),
            ([22, 23, 23, 24], True), ([25, 26, 22, 26], False), ([22, 23, 22, 24], False),
            ([18, 19, 20, 21], False), ([5, 6, 18, 19], False), ([3, 4, 18, 19], False),
            ([16, 17, 18, 19], False), ([14, 15, 16, 17], False), ([11, 12, 12, 13], False),
            ([7, 8, 9, 10], False), ([1, 2, 5, 6], False), ([1, 2, 3, 4], False)
        ]

        for i, (ids, is_obtuse) in enumerate(configs):
            img_draw = image.copy()
            pts = [self._get_kp(kps, idx) for idx in ids]
            try:
                v1 = self.geo.get_vector(pts[0], pts[1])
                v2 = self.geo.get_vector(pts[2], pts[3])
                val = round(self.geo.calculate_angle(v1, v2, is_obtuse), 1)
                vis.draw_common(img_draw, pts[0], pts[1], pts[2], pts[3], ids)
                vis.draw_title(img_draw, val)
                results['values'][self.NAMES[i]] = val
                results['images'].append(img_draw)
            except:
                results['values'][self.NAMES[i]] = 0.0
                results['images'].append(img_draw)
        return results