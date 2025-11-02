import numpy as np
from Node import Node

class KD_tree_a:
    def __init__(self, points, depth=0):
        if len(points) == 0:
            self.node = None
            return
        k = points.shape[1]
        axis = depth % k
        points = points[points[:, axis].argsort()]
        median = len(points) // 2
        self.node = Node(
            point=points[median],
            axis=axis,
            left=KD_tree_a(points[:median], depth + 1).node,
            right=KD_tree_a(points[median + 1:], depth + 1).node
        )

    def insert(self, root, point, depth=0):
        if root is None:
            return Node(point, depth % len(point))
        axis = root.axis
        if point[axis] < root.point[axis]:
            root.left = self.insert(root.left, point, depth + 1)
        else:
            root.right = self.insert(root.right, point, depth + 1)
        return root

    def nearest(self, root, target, depth=0, best=None, best_dist=float("inf")):
        if root is None:
            return best, best_dist
        axis = root.axis
        dist = np.linalg.norm(target - root.point)
        if dist < best_dist:
            best, best_dist = root.point, dist
        diff = target[axis] - root.point[axis]
        near, far = (root.left, root.right) if diff < 0 else (root.right, root.left)
        best, best_dist = self.nearest(near, target, depth + 1, best, best_dist)
        if abs(diff) < best_dist:
            best, best_dist = self.nearest(far, target, depth + 1, best, best_dist)
        return best, best_dist