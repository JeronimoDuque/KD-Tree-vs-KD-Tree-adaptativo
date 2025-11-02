# visual_tkinter.py
# Visualiza el KD-Tree espacialmente con Tkinter Canvas
import tkinter as tk
import numpy as np
import sys

def draw_kdtree(canvas, node, x0, y0, x1, y1, width=600, depth=0):
    """Dibuja recursivamente las divisiones del KD-Tree."""
    if node is None:
        return []
    def mx(x): return int(x * width)
    def my(y): return int((1 - y) * width)

    x, y = node.point
    axis = node.axis
    hyperplanes = []

    if axis == 0:  # división vertical (x)
        canvas.create_line(mx(x), my(y0), mx(x), my(y1), dash=(4, 4), width=1.5, fill='red')
        canvas.create_text(mx(x)+20, my((y0+y1)/2), text=f"x={x:.2f}", fill='red', font=('Arial', 8))
        hyperplanes.append({'axis': 'x', 'coord': x, 'depth': depth, 'bbox': (x0, y0, x1, y1)})
        hyperplanes += draw_kdtree(canvas, node.left, x0, y0, x, y1, width, depth+1)
        hyperplanes += draw_kdtree(canvas, node.right, x, y0, x1, y1, width, depth+1)
    else:  # división horizontal (y)
        canvas.create_line(mx(x0), my(y), mx(x1), my(y), dash=(4, 4), width=1.5, fill='blue')
        canvas.create_text(mx((x0+x1)/2), my(y)-15, text=f"y={y:.2f}", fill='blue', font=('Arial', 8))
        hyperplanes.append({'axis': 'y', 'coord': y, 'depth': depth, 'bbox': (x0, y0, x1, y1)})
        hyperplanes += draw_kdtree(canvas, node.left, x0, y0, x1, y, width, depth+1)
        hyperplanes += draw_kdtree(canvas, node.right, x0, y, x1, y1, width, depth+1)

    # Dibujar el punto
    r = 3
    canvas.create_oval(mx(x)-r, my(y)-r, mx(x)+r, my(y)+r, fill='black')
    return hyperplanes


def show_tree(root, points=None, new_points=None, width=700):
    root_win = tk.Tk()
    root_win.title("Visualización del KD-Tree (Tkinter)")
    canvas = tk.Canvas(root_win, width=width, height=width, bg='white')
    canvas.pack()

    def draw_points(pts, color):
        if pts is None:
            return
        for p in pts:
            xpix = int(p[0]*width)
            ypix = int((1-p[1])*width)
            r = 4
            canvas.create_oval(xpix-r, ypix-r, xpix+r, ypix+r, fill=color, outline='')

    draw_points(points, 'green')
    draw_points(new_points, 'orange')

    hyperplanes = draw_kdtree(canvas, root, 0, 0, 1, 1, width)
    print("Hiperplanos (eje, coord, profundidad):")
    for h in hyperplanes[:15]:
        print(f"  eje={h['axis']} coord={h['coord']:.3f} depth={h['depth']}")
    root_win.mainloop()


if __name__ == "__main__":
    # Si se ejecuta directamente desde main.py (subprocess)
    try:
        from KD_tree_a import KD_tree_a
        import numpy as np
        np.random.seed(42)
        pts = np.random.rand(30, 2)
        tree = KD_tree_a(pts)
        root = tree.node
        points = pts
        new_points = None
    except Exception as e:
        print("⚠️ Error al crear árbol para visualización:", e)
        sys.exit(1)

    show_tree(root, points, new_points, width=700)