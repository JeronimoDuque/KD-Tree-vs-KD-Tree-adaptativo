import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from KD_tree import KD_tree
from KD_tree_a import KD_tree_a


#  Generación de datos

np.random.seed(42)  # semilla fija para reproducibilidad
points = np.random.rand(1000, 2)       # 1000 puntos iniciales
new_points = np.random.rand(1000, 2)   # 1000 nuevos puntos a insertar
query_points = np.random.rand(1000, 2) # 1000 puntos de consulta


#  Construcción inicial de árboles

start_c = time.perf_counter()
classic_tree = KD_tree(points)
end_c = time.perf_counter()

start_a = time.perf_counter()
adaptive_tree = KD_tree_a(points)
end_a = time.perf_counter()


#  Inserción de 1000 nuevos puntos

insert_times_classic = []
insert_times_adaptive = []

# KD-Tree clásico → requiere reconstrucción total
for i in range(1000):
    start = time.perf_counter()
    updated_points = np.vstack((points, new_points[:i+1]))
    KD_tree(updated_points)
    end = time.perf_counter()
    insert_times_classic.append(end - start)

# KD-Tree adaptativo → inserción incremental
for i in range(1000):
    start = time.perf_counter()
    adaptive_tree.node = adaptive_tree.insert(adaptive_tree.node, new_points[i])
    end = time.perf_counter()
    insert_times_adaptive.append(end - start)


#  Búsqueda de vecinos más cercanos (KNN)

search_times_classic = []
search_times_adaptive = []

for q in query_points:
    start = time.perf_counter()
    classic_tree.nearest(classic_tree.node, q)
    end = time.perf_counter()
    search_times_classic.append(end - start)

for q in query_points:
    start = time.perf_counter()
    adaptive_tree.nearest(adaptive_tree.node, q)
    end = time.perf_counter()
    search_times_adaptive.append(end - start)


#  Cálculos de tiempo promedio

build_time_c = end_c - start_c
build_time_a = end_a - start_a
mean_insert_c = np.mean(insert_times_classic)
mean_insert_a = np.mean(insert_times_adaptive)
mean_search_c = np.mean(search_times_classic)
mean_search_a = np.mean(search_times_adaptive)


#  Visualización con Tkinter

print("\n=== Mostrando visualización del árbol (Tkinter) ===")
try:
    from visual_tkinter import show_tree
    show_tree(adaptive_tree.node, points, new_points, width=700)
    print(" Visualización Tkinter mostrada correctamente.")
except Exception as e:
    print(" No se pudo mostrar la visualización Tkinter:", e)


#  Reporte de resultados

print("\n=== COMPARACIÓN DE RENDIMIENTO ===")
print(f"Construcción Clásico:    {build_time_c:.6f} s")
print(f"Construcción Adaptativo: {build_time_a:.6f} s")
print(f"Inserción promedio Clásico: {mean_insert_c*1000:.6f} ms")
print(f"Inserción promedio Adaptativo: {mean_insert_a*1000:.6f} ms")
print(f"Búsqueda promedio Clásico: {mean_search_c*1000:.6f} ms")
print(f"Búsqueda promedio Adaptativo: {mean_search_a*1000:.6f} ms")


#  Gráficas de rendimiento

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(insert_times_classic, label="KD-Tree Clásico", color='red')
plt.plot(insert_times_adaptive, label="KD-Tree Adaptativo", color='green')
plt.title("Tiempo de Inserción (1000 puntos)")
plt.xlabel("Inserción #")
plt.ylabel("Tiempo (s)")
plt.legend()

plt.subplot(1,2,2)
plt.plot(search_times_classic, label="KD-Tree Clásico", color='blue')
plt.plot(search_times_adaptive, label="KD-Tree Adaptativo", color='orange')
plt.title("Tiempo de Búsqueda (1000 consultas)")
plt.xlabel("Consulta #")
plt.ylabel("Tiempo (s)")
plt.legend()

plt.tight_layout()
plt.show()

print("\n=== Ejecutando actividad comparativa KD-Tree ===")

try:
    import Actividad_4
    print(" Actividad KD-Tree ejecutada correctamente.")
except Exception as e:
    print(" No se pudo ejecutar la actividad KD-Tree:", e)