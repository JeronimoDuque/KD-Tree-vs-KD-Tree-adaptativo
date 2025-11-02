import numpy as np
from KD_tree import KD_tree
from KD_tree_a import KD_tree_a
from visual_tkinter import show_tree


#  Generar 50 puntos aleatorios en 2D

np.random.seed(42)
points = np.random.rand(50, 2)


#  Construir KD-Tree clásico y adaptativo

classic_tree = KD_tree(points)
adaptive_tree = KD_tree_a(points)

print(" Árboles iniciales construidos con 50 puntos.")


# Insertar 10 nuevos puntos

new_points = np.random.rand(10, 2)

# Clásico → reconstruir con los nuevos puntos
classic_updated = KD_tree(np.vstack((points, new_points)))

# Adaptativo → insertar dinámicamente
for p in new_points:
    adaptive_tree.node = adaptive_tree.insert(adaptive_tree.node, p)


# Mostrar visualmente cómo cambia cada árbol

print("\n=== Visualización del KD-Tree clásico (después de inserciones) ===")
show_tree(classic_updated.node, np.vstack((points, new_points)))

print("\n=== Visualización del KD-Tree adaptativo (después de inserciones) ===")
show_tree(adaptive_tree.node, np.vstack((points, new_points)))


# Conclusión

print("\n=== CONCLUSIÓN ===")
print(
    "El KD-Tree adaptativo es más adecuado para aplicaciones en tiempo real, "
    "ya que puede ajustarse cuando se agregan nuevos puntos sin tener que "
    "reconstruir toda la estructura. El KD-Tree clásico, en cambio, requiere "
    "una reconstrucción completa cada vez que los datos cambian."
)