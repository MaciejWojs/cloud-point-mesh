#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import open3d as o3d
import time
from functools import wraps, lru_cache
import os
import json
import argparse
from multiprocessing import Pool, cpu_count

# Opcjonalne biblioteki
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba niedostępny - używam standardowego NumPy")
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("NetworkX niedostępny - używam prostej optymalizacji")

# ---------------------
# Konfiguracja
# ---------------------
class Config:
    OUTPUT_DIR = "./output"
    VOXEL_SIZE = 0.002  # Zwiększono z 0.001 dla dużych zbiorów
    POISSON_DEPTH = 7  # Bezpieczna wartość
    VISIBILITY_THRESHOLD = 0.1
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080
    DRAWING_SPEED = 100  # mm/s
    TRAVEL_SPEED = 150  # mm/s
    Z_LIFT = 5  # mm
    USE_PARALLEL = True
    MAX_WORKERS = min(4, cpu_count())
    # Nowe parametry dla jakości
    MIN_TRIANGLES = 15000  # Minimum trójkątów w finalnym meshu
    MAX_TRIANGLES = 60000  # Maximum trójkątów w finalnym meshu
    SMOOTHING_ITERATIONS = 2  # Liczba iteracji wygładzania
    MAX_POINTS_FOR_ORIENT = 100000  # Max punktów dla orient_normals (qhull limit)

# ---------------------
# Dekorator mierzący czas
# ---------------------
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[BENCH] {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

# ---------------------
# Loader chmur punktów (uproszczony)
# ---------------------
@measure_time
def load_point_cloud(path):
    """Wczytuje chmurę punktów"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plik nie istnieje: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    print(f"Wczytywanie {path} (format: {ext})")
    
    if ext in (".las", ".laz"):
        import laspy
        las = laspy.open(path, mode='r')
        points = las.read()
        xyz = np.vstack((points.x, points.y, points.z)).T.astype(np.float32)
        return xyz
    else:
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            raise ValueError("Plik nie zawiera punktów")
        return np.asarray(pcd.points, dtype=np.float32)

# ---------------------
# Przetwarzanie chmury punktów (INTELIGENTNE - NAPRAWIONE)
# ---------------------
@measure_time
def preprocess_point_cloud(points, voxel_size=0.002):
    """Inteligentne czyszczenie z adaptywnym downsamplingiem"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"Punkty początkowe: {len(points)}")
    
    # 1. Outlier removal - bardzo ważne dla jakości
    if len(pcd.points) > 2000:
        print("Usuwanie outlierów (pass 1)...")
        pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)  # Zmniejszono z 30
        removed = len(points) - len(inliers)
        if removed > 0:
            print(f"  Usunięto {removed} outlierów ({removed/len(points)*100:.1f}%)")
    
    # 2. Adaptywny downsampling - zachowuje więcej punktów w obszarach z detalami
    initial_count = len(pcd.points)
    
    # Dla ogromnych zbiorów (>1M), zwiększ target
    if initial_count > 1000000:
        target_count = min(80000, initial_count)  # Zwiększono dla dużych zbiorów
        print(f"Duża chmura wykryta ({initial_count} punktów)")
    else:
        target_count = min(50000, initial_count)
    
    if initial_count > target_count:
        print(f"Adaptywny downsampling: {initial_count} -> ~{target_count} punktów")
        
        # Dla ogromnych zbiorów - pomijamy analizę gęstości (za wolna)
        if initial_count > 500000:
            # Prosta metoda - użyj zwiększonego voxel size
            actual_voxel = voxel_size * 1.5
            print(f"  Użyto voxel size: {actual_voxel:.4f} (uproszczona metoda dla dużego zbioru)")
            pcd = pcd.voxel_down_sample(actual_voxel)
        else:
            # Oblicz gęstość lokalną - tylko dla mniejszych zbiorów
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            points_array = np.asarray(pcd.points)
            densities = []
            
            sample_size = min(500, len(points_array))  # Zmniejszono z 1000
            sample_indices = np.random.choice(len(points_array), sample_size, replace=False)
            
            for idx in sample_indices:
                [k, idx_vec, dist_vec] = pcd_tree.search_radius_vector_3d(pcd.points[idx], voxel_size * 2)
                densities.append(k)
            
            avg_density = np.mean(densities)
            
            # Adaptywny voxel size na podstawie gęstości
            if avg_density > 50:
                actual_voxel = voxel_size * 1.3
            elif avg_density < 20:
                actual_voxel = voxel_size * 0.9
            else:
                actual_voxel = voxel_size
            
            print(f"  Użyto voxel size: {actual_voxel:.4f} (gęstość lokalna: {avg_density:.1f})")
            pcd = pcd.voxel_down_sample(actual_voxel)
    else:
        pcd = pcd.voxel_down_sample(voxel_size)
    
    print(f"Po downsamplingu: {len(pcd.points)} punktów")
    
    # 3. Drugi pass outlier removal - tylko jeśli nie za duży
    if 5000 < len(pcd.points) < 200000:  # Dodano górny limit
        print("Usuwanie outlierów (pass 2)...")
        pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.5)
        removed = len(pcd.points) - len(inliers)
        if removed > 0:
            print(f"  Usunięto {removed} dodatkowych outlierów")
    elif len(pcd.points) >= 200000:
        print(f"Zbyt dużo punktów ({len(pcd.points)}), pomijam drugi pass outlier removal")
    
    return np.asarray(pcd.points, dtype=np.float32)

# ---------------------
# Budowa meshu (ULTRA JAKOŚĆ + OPTYMALIZACJA - NAPRAWIONY QHULL)
# ---------------------
@measure_time
def build_mesh(points, depth=7):
    """Buduje mesh z maksymalną jakością przy minimalnej liczbie trójkątów"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    num_points = len(pcd.points)
    print(f"Budowanie meshu z {num_points} punktów...")
    
    # 1. Estymacja normalnych - najlepsze parametry
    print("Estymacja normalnych (wysokiej jakości)...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # 2. Orientacja normalnych - POMIŃ DLA DUŻYCH ZBIORÓW (qhull problem)
    if num_points < Config.MAX_POINTS_FOR_ORIENT:
        print("Orientacja normalnych...")
        try:
            pcd.orient_normals_consistent_tangent_plane(30)  # Zmniejszono z 50
        except RuntimeError as e:
            print(f"  UWAGA: Błąd orientacji normalnych: {e}")
            print("  Kontynuuję bez orientacji (może wpłynąć na jakość)")
    else:
        print(f"  POMINIĘTO orientację normalnych (za dużo punktów: {num_points})")
        print("  Używam prostej orientacji...")
        # Prosta orientacja - wszystkie normalne w jednym kierunku
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    
    # 3. Poisson Reconstruction z optymalnymi parametrami
    print(f"Poisson reconstruction (depth={depth}, high quality)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, 
        depth=depth,
        width=0,  # Auto
        scale=1.1,
        linear_fit=False,
        n_threads=Config.MAX_WORKERS
    )
    
    print(f"  Wygenerowano: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów")
    
    # 4. Inteligentne filtrowanie gęstości
    if len(densities) > 0:
        densities = np.asarray(densities)
        
        # Użyj percentyla zamiast stałego progu
        density_threshold = np.percentile(densities, 2)  # Zwiększono z 1% -> 2% (mniej agresywne)
        
        vertices_to_keep = densities > density_threshold
        mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
        print(f"  Po filtracji gęstości: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów")
    
    # 5. Czyszczenie geometrii
    print("Czyszczenie geometrii...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    
    print(f"  Po czyszczeniu: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów")
    
    # 6. Inteligentne upraszczanie - zachowaj geometrię
    current_triangles = len(mesh.triangles)
    
    if current_triangles > Config.MAX_TRIANGLES:
        target = Config.MAX_TRIANGLES
        print(f"Upraszczanie (adaptywne): {current_triangles} -> {target} trójkątów...")
        mesh = mesh.simplify_quadric_decimation(target)
        
    elif current_triangles > Config.MIN_TRIANGLES * 2:
        target = max(Config.MIN_TRIANGLES, int(current_triangles * 0.75))  # Mniej agresywne (0.7 -> 0.75)
        print(f"Upraszczanie (delikatne): {current_triangles} -> {target} trójkątów...")
        mesh = mesh.simplify_quadric_decimation(target)
    else:
        print(f"Mesh optymalny, pomijam upraszczanie ({current_triangles} trójkątów)")
    
    # 7. Wygładzanie - zachowuje cechy, usuwa szum
    if Config.SMOOTHING_ITERATIONS > 0:
        print(f"Wygładzanie Laplacian ({Config.SMOOTHING_ITERATIONS} iteracji)...")
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=Config.SMOOTHING_ITERATIONS,
            lambda_filter=0.5
        )
    
    # 8. Oblicz normalne
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    # 9. Podsumowanie jakości
    print(f"\n{'='*50}")
    print(f"FINALNY MESH - PODSUMOWANIE:")
    print(f"  • Wierzchołki: {len(mesh.vertices):,}")
    print(f"  • Trójkąty: {len(mesh.triangles):,}")
    print(f"  • Bbox: {mesh.get_axis_aligned_bounding_box().get_extent()}")
    
    try:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        triangle_areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        print(f"  • Średni rozmiar trójkąta: {np.mean(triangle_areas):.6f}")
        print(f"  • Min/Max obszar: {np.min(triangle_areas):.6f} / {np.max(triangle_areas):.6f}")
        print(f"  • Całkowita powierzchnia: {np.sum(triangle_areas):.2f}")
    except Exception as e:
        print(f"  • Nie można obliczyć statystyk obszarów: {e}")
    
    try:
        if mesh.is_watertight():
            print(f"  • Status: WATERTIGHT ✓")
        else:
            print(f"  • Status: NON-WATERTIGHT")
    except:
        print(f"  • Status: Nie można określić")
    
    print(f"{'='*50}\n")
    
    return mesh

# ---------------------
# Numba-zoptymalizowane funkcje pomocnicze
# ---------------------
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def compute_backface_culling_numba(centers, normals, camera_pos, threshold):
        """Backface culling z Numba - super szybkie"""
        n = len(centers)
        visible = np.zeros(n, dtype=np.bool_)
        
        for i in prange(n):
            view_dir = camera_pos - centers[i]
            view_len = np.sqrt(view_dir[0]**2 + view_dir[1]**2 + view_dir[2]**2)
            if view_len > 0:
                view_dir = view_dir / view_len
                dot_prod = normals[i, 0]*view_dir[0] + normals[i, 1]*view_dir[1] + normals[i, 2]*view_dir[2]
                visible[i] = dot_prod > threshold
        
        return visible

    @jit(nopython=True, cache=True)
    def find_nearest_numba(current_end, starts, ends, used_mask):
        """Znajdź najbliższą linię - Numba version"""
        n = len(starts)
        best_idx = -1
        best_dist = np.inf
        flip = False
        
        for i in range(n):
            if used_mask[i]:
                continue
            
            # Odległość do startu
            dx = current_end[0] - starts[i, 0]
            dy = current_end[1] - starts[i, 1]
            dz = current_end[2] - starts[i, 2]
            dist_start = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist_start < best_dist:
                best_dist = dist_start
                best_idx = i
                flip = False
            
            # Odległość do końca
            dx = current_end[0] - ends[i, 0]
            dy = current_end[1] - ends[i, 1]
            dz = current_end[2] - ends[i, 2]
            dist_end = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist_end < best_dist:
                best_dist = dist_end
                best_idx = i
                flip = True
        
        return best_idx, flip

# ---------------------
# Generowanie linii 3D (zoptymalizowane z Numba)
# ---------------------
@measure_time
def generate_lines_from_mesh(mesh, camera_position, threshold=0.1):
    """Generuje linie z widocznych krawędzi - NUMBA ACCELERATED"""
    if len(mesh.triangles) == 0:
        return []

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    
    if len(mesh.triangle_normals) == 0:
        mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals, dtype=np.float32)
    
    # Backface culling - użyj Numba jeśli dostępne
    centers = np.mean(vertices[triangles], axis=1).astype(np.float32)
    
    if NUMBA_AVAILABLE:
        visible_mask = compute_backface_culling_numba(centers, normals, camera_position.astype(np.float32), threshold)
        visible_triangles = triangles[visible_mask]
    else:
        # Fallback bez Numba
        view_dirs = camera_position - centers
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)
        dot_products = np.einsum('ij,ij->i', normals, view_dirs)
        visible_triangles = triangles[dot_products > threshold]
    
    if len(visible_triangles) == 0:
        return []
    
    # Zbierz krawędzie - zoptymalizowane
    edges = set()
    for tri in visible_triangles:
        edges.add(tuple(sorted([tri[0], tri[1]])))
        edges.add(tuple(sorted([tri[1], tri[2]])))
        edges.add(tuple(sorted([tri[2], tri[0]])))
    
    lines_3d = [(vertices[e[0]], vertices[e[1]]) for e in edges]
    print(f"Wygenerowano {len(lines_3d)} linii 3D")
    return lines_3d

# ---------------------
# Optymalizacja ścieżki (NUMBA ACCELERATED)
# ---------------------
@measure_time
def optimize_path(lines_3d, max_lines=2000):
    """Zoptymalizowana wersja z Numba - ULTRA SZYBKA"""
    if not lines_3d or len(lines_3d) <= 1:
        return lines_3d
    
    # Ogranicz liczbę linii
    if len(lines_3d) > max_lines:
        indices = np.linspace(0, len(lines_3d)-1, max_lines, dtype=int)
        lines_3d = [lines_3d[i] for i in indices]
    
    print(f"Optymalizacja {len(lines_3d)} linii...")
    
    # Konwersja do numpy
    starts = np.array([line[0] for line in lines_3d], dtype=np.float32)
    ends = np.array([line[1] for line in lines_3d], dtype=np.float32)
    
    optimized = []
    used = np.zeros(len(lines_3d), dtype=np.bool_)
    
    # Rozpocznij od pierwszej linii
    current_idx = 0
    used[current_idx] = True
    optimized.append(lines_3d[current_idx])
    current_end = ends[current_idx]
    
    # Greedy nearest neighbor
    if NUMBA_AVAILABLE:
        # Użyj Numba dla szybkości
        for _ in range(len(lines_3d) - 1):
            best_idx, flip = find_nearest_numba(current_end, starts, ends, used)
            
            if best_idx == -1:
                break
            
            used[best_idx] = True
            next_line = lines_3d[best_idx]
            
            if flip:
                next_line = (next_line[1], next_line[0])
                current_end = next_line[1]
            else:
                current_end = ends[best_idx]
            
            optimized.append(next_line)
    else:
        # Fallback bez Numba (wolniejszy)
        for _ in range(len(lines_3d) - 1):
            unused_indices = np.where(~used)[0]
            if len(unused_indices) == 0:
                break
            
            dist_to_starts = np.linalg.norm(starts[unused_indices] - current_end, axis=1)
            dist_to_ends = np.linalg.norm(ends[unused_indices] - current_end, axis=1)
            
            min_start_idx = np.argmin(dist_to_starts)
            min_end_idx = np.argmin(dist_to_ends)
            
            if dist_to_starts[min_start_idx] < dist_to_ends[min_end_idx]:
                best_local_idx = min_start_idx
                flip = False
            else:
                best_local_idx = min_end_idx
                flip = True
            
            best_idx = unused_indices[best_local_idx]
            used[best_idx] = True
            
            next_line = lines_3d[best_idx]
            if flip:
                next_line = (next_line[1], next_line[0])
            
            optimized.append(next_line)
            current_end = next_line[1]
    
    print(f"Zoptymalizowano {len(optimized)} linii")
    return optimized

# ---------------------
# Alternatywna szybsza optymalizacja (k-d tree) - NAPRAWIONA
# ---------------------
@measure_time
def optimize_path_kdtree(lines_3d, max_lines=2000):
    """Optymalizacja z k-d tree - NAPRAWIONA nieskończona pętla"""
    if not lines_3d or len(lines_3d) <= 1:
        return lines_3d
    
    # Ogranicz liczbę linii
    if len(lines_3d) > max_lines:
        indices = np.linspace(0, len(lines_3d)-1, max_lines, dtype=int)
        lines_3d = [lines_3d[i] for i in indices]
    
    from scipy.spatial import cKDTree
    
    print(f"Optymalizacja {len(lines_3d)} linii (k-d tree)...")
    
    # Przygotuj punkty
    all_points = []
    point_to_line = []
    for idx, (start, end) in enumerate(lines_3d):
        all_points.append(start)
        all_points.append(end)
        point_to_line.extend([idx, idx])
    
    all_points = np.array(all_points, dtype=np.float32)
    point_to_line = np.array(point_to_line, dtype=np.int32)
    
    # Zbuduj k-d tree
    tree = cKDTree(all_points)
    
    optimized = []
    used = set()
    
    current = lines_3d[0]
    optimized.append(current)
    used.add(0)
    current_end = current[1]
    
    # Greedy z k-d tree - z zabezpieczeniami
    max_iterations = len(lines_3d)  # WAŻNE: limit iteracji
    iteration = 0
    
    while len(used) < len(lines_3d) and iteration < max_iterations:
        iteration += 1
        
        # Znajdź k najbliższych punktów
        k_neighbors = min(30, len(all_points))  # Zwiększono z 20 -> 30
        distances, indices = tree.query(current_end, k=k_neighbors)
        
        # Upewnij się że to tablice (nie skalary)
        if np.isscalar(distances):
            distances = np.array([distances])
            indices = np.array([indices])
        
        # Znajdź najbliższą nieużytą linię
        found = False
        for dist, idx in zip(distances, indices):
            line_idx = point_to_line[idx]
            if line_idx not in used:
                used.add(line_idx)
                next_line = lines_3d[line_idx]
                
                # Sprawdź orientację
                dist_to_start = np.linalg.norm(next_line[0] - current_end)
                dist_to_end = np.linalg.norm(next_line[1] - current_end)
                
                if dist_to_start > dist_to_end:
                    next_line = (next_line[1], next_line[0])
                
                optimized.append(next_line)
                current_end = next_line[1]
                found = True
                break
        
        # Jeśli nie znaleziono, coś jest nie tak - przerwij
        if not found:
            print(f"UWAGA: Nie można znaleźć następnej linii po {iteration} iteracjach")
            break
    
    if iteration >= max_iterations:
        print(f"UWAGA: Osiągnięto limit iteracji ({max_iterations})")
    
    print(f"Zoptymalizowano {len(optimized)} linii (iteracji: {iteration})")
    return optimized

# ---------------------
# Renderowanie (naprawione)
# ---------------------
@measure_time
def render_mesh(mesh, camera_pos, look_at, output_path, lines_3d=None):
    """Renderuje mesh do obrazu z fallback do wireframe"""
    try:
        # Próba Open3D rendering
        os.environ['OPEN3D_CPU_RENDERING'] = 'true'
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=Config.IMAGE_WIDTH, height=Config.IMAGE_HEIGHT, visible=False)
        
        # Dodaj mesh
        mesh_copy = o3d.geometry.TriangleMesh(mesh)
        mesh_copy.compute_vertex_normals()
        mesh_copy.paint_uniform_color([0.7, 0.7, 0.7])
        vis.add_geometry(mesh_copy)
        
        # Dodaj linie
        if lines_3d:
            points = []
            line_indices = []
            for i, (start, end) in enumerate(lines_3d[:3000]):  # Limit
                points.extend([start, end])
                line_indices.append([i*2, i*2+1])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.paint_uniform_color([1, 0, 0])
            vis.add_geometry(line_set)
        
        # Ustaw kamerę
        ctr = vis.get_view_control()
        if ctr is None:
            raise RuntimeError("View control failed")
            
        vis.poll_events()
        vis.update_renderer()
        
        forward = (look_at - camera_pos) / np.linalg.norm(look_at - camera_pos)
        up = np.array([0, 0, 1]) if abs(np.dot(forward, [0, 0, 1])) < 0.9 else np.array([0, 1, 0])
        
        ctr.set_lookat(look_at)
        ctr.set_front(forward)
        ctr.set_up(up)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Zapisz obraz
        image = vis.capture_screen_float_buffer(do_render=True)
        image_array = np.asarray(image)
        
        if image_array.size == 0:
            raise RuntimeError("Empty render buffer")
        
        image_bgr = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        print(f"Render 3D zapisany: {output_path}")
        
        vis.destroy_window()
        return output_path
        
    except Exception as e:
        print(f"Błąd renderowania 3D: {e}")
        print("Przełączam na wireframe 2D...")
        return render_wireframe_2d(mesh, camera_pos, look_at, output_path, lines_3d)

def render_wireframe_2d(mesh, camera_pos, look_at, output_path, lines_3d=None):
    """Fallback: renderowanie wireframe 2D"""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Tło z gradientem
    img = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3), dtype=np.uint8)
    for y in range(Config.IMAGE_HEIGHT):
        gray = int(30 + (60 - 30) * (y / Config.IMAGE_HEIGHT))
        img[y, :] = [gray, gray, gray]
    
    # Projekcja
    view_dir = (look_at - camera_pos) / np.linalg.norm(look_at - camera_pos)
    
    # Wybierz osie projekcji
    if abs(view_dir[2]) > 0.7:  # Z góry/dołu
        proj_vertices = vertices[:, :2]
        axis_labels = ['X', 'Y']
    elif abs(view_dir[1]) > 0.7:  # Z przodu/tyłu
        proj_vertices = vertices[:, [0, 2]]
        axis_labels = ['X', 'Z']
    else:  # Z boku
        proj_vertices = vertices[:, [1, 2]]
        axis_labels = ['Y', 'Z']
    
    # Usuń NaN/Inf
    valid_mask = np.isfinite(proj_vertices).all(axis=1)
    proj_vertices_clean = proj_vertices[valid_mask]
    
    if len(proj_vertices_clean) == 0:
        # Fallback do tekstu
        cv2.putText(img, "No valid vertices", (50, Config.IMAGE_HEIGHT//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imwrite(output_path, img)
        return output_path
    
    # Skalowanie
    min_vals = np.min(proj_vertices_clean, axis=0)
    max_vals = np.max(proj_vertices_clean, axis=0)
    ranges = max_vals - min_vals
    ranges = np.where(ranges < 1e-6, 1.0, ranges)
    
    margin = 0.1
    scale = min(Config.IMAGE_WIDTH * (1-2*margin) / ranges[0], 
                Config.IMAGE_HEIGHT * (1-2*margin) / ranges[1])
    
    center = (min_vals + max_vals) / 2
    scaled_vertices = (proj_vertices - center) * scale
    scaled_vertices[:, 0] += Config.IMAGE_WIDTH // 2
    scaled_vertices[:, 1] += Config.IMAGE_HEIGHT // 2
    scaled_vertices = scaled_vertices.astype(int)
    
    # Rysuj trójkąty wireframe
    triangle_count = 0
    for tri in triangles:
        if all(valid_mask[idx] for idx in tri):
            pts = scaled_vertices[tri]
            if all(0 <= p[0] < Config.IMAGE_WIDTH and 0 <= p[1] < Config.IMAGE_HEIGHT for p in pts):
                cv2.polylines(img, [pts], True, (180, 180, 180), 1, cv2.LINE_AA)
                triangle_count += 1
    
    # Rysuj linie robota
    if lines_3d:
        line_count = 0
        for start_3d, end_3d in lines_3d[:5000]:  # Limit
            # Projekcja linii
            if abs(view_dir[2]) > 0.7:
                start_2d = start_3d[:2]
                end_2d = end_3d[:2]
            elif abs(view_dir[1]) > 0.7:
                start_2d = start_3d[[0, 2]]
                end_2d = end_3d[[0, 2]]
            else:
                start_2d = start_3d[[1, 2]]
                end_2d = end_3d[[1, 2]]
            
            # Skalowanie
            start_scaled = ((start_2d - center) * scale + [Config.IMAGE_WIDTH//2, Config.IMAGE_HEIGHT//2]).astype(int)
            end_scaled = ((end_2d - center) * scale + [Config.IMAGE_WIDTH//2, Config.IMAGE_HEIGHT//2]).astype(int)
            
            if (0 <= start_scaled[0] < Config.IMAGE_WIDTH and 0 <= start_scaled[1] < Config.IMAGE_HEIGHT and
                0 <= end_scaled[0] < Config.IMAGE_WIDTH and 0 <= end_scaled[1] < Config.IMAGE_HEIGHT):
                cv2.line(img, tuple(start_scaled), tuple(end_scaled), (0, 0, 255), 2, cv2.LINE_AA)
                line_count += 1
        
        print(f"Narysowano {line_count} linii robota")
    
    # Panel informacyjny
    cv2.rectangle(img, (0, 0), (400, 120), (20, 20, 20), -1)
    cv2.putText(img, f"Wireframe ({axis_labels[0]}-{axis_labels[1]})", (15, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Vertices: {len(vertices):,}", (15, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"Triangles: {triangle_count:,}/{len(triangles):,}", (15, 75), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if lines_3d:
        cv2.putText(img, f"Robot lines: {len(lines_3d):,}", (15, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    print(f"Narysowano {triangle_count} trójkątów wireframe")
    cv2.imwrite(output_path, img)
    print(f"Wireframe 2D zapisany: {output_path}")
    return output_path

# ---------------------
# Eksport dla robota Hanwha HCR-3A (zamiast G-code)
# ---------------------
@measure_time
def export_hanwha_commands(lines_3d, output_dir):
    """Generuje komendy dla robota Hanwha HCR-3A"""
    if not lines_3d:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    commands_file = os.path.join(output_dir, "hanwha_robot_commands.txt")
    
    with open(commands_file, 'w') as f:
        f.write("# Komendy dla robota Hanwha HCR-3A\n")
        f.write(f"# Wygenerowano: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Liczba linii: {len(lines_3d)}\n\n")
        
        # Inicjalizacja
        f.write("ROBOT_INIT\n")
        f.write(f"SET_SPEED DRAW={Config.DRAWING_SPEED} TRAVEL={Config.TRAVEL_SPEED}\n")
        f.write(f"SET_Z_LIFT {Config.Z_LIFT}\n")
        f.write("TOOL_ON\n\n")
        
        # Pierwsza pozycja
        first_start = lines_3d[0][0]
        f.write("# Przejście do pozycji startowej\n")
        f.write(f"MOVE_SAFE {first_start[0]*1000:.3f} {first_start[1]*1000:.3f} {first_start[2]*1000:.3f}\n")
        f.write("PEN_DOWN\n\n")
        
        # Rysowanie linii
        f.write("# Rozpoczęcie rysowania\n")
        for i, (start, end) in enumerate(lines_3d, 1):
            f.write(f"# Linia {i}/{len(lines_3d)}\n")
            
            # Sprawdź czy trzeba przemieścić narzędzie
            if i > 1:
                prev_end = lines_3d[i-2][1]
                if not np.allclose(start, prev_end, atol=0.001):
                    f.write("PEN_UP\n")
                    f.write(f"MOVE_FAST {start[0]*1000:.3f} {start[1]*1000:.3f} {start[2]*1000:.3f}\n")
                    f.write("PEN_DOWN\n")
            
            # Rysuj linię
            f.write(f"DRAW_LINE {end[0]*1000:.3f} {end[1]*1000:.3f} {end[2]*1000:.3f}\n")
        
        # Zakończenie
        f.write("\n# Zakończenie\n")
        f.write("PEN_UP\n")
        f.write("MOVE_HOME\n")
        f.write("TOOL_OFF\n")
        f.write("ROBOT_END\n")
    
    print(f"Komendy Hanwha zapisane: {commands_file}")
    return commands_file

# ---------------------
# Eksport danych robota (zaktualizowany - NAPRAWIONY JSON)
# ---------------------
@measure_time
def export_robot_data(lines_3d, output_dir):
    """Eksportuje dane dla robota"""
    if not lines_3d:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Punkty (milimetry)
    points = np.vstack([[s*1000, e*1000] for s, e in lines_3d])
    np.savetxt(os.path.join(output_dir, "robot_points_mm.txt"), points, fmt="%.3f")
    
    # Linie (milimetry)
    with open(os.path.join(output_dir, "robot_lines_mm.txt"), 'w') as f:
        f.write("# Format: start_x start_y start_z end_x end_y end_z (wszystko w mm)\n")
        for s, e in lines_3d:
            f.write(f"{s[0]*1000:.3f} {s[1]*1000:.3f} {s[2]*1000:.3f} {e[0]*1000:.3f} {e[1]*1000:.3f} {e[2]*1000:.3f}\n")
    
    # Statystyki - konwertuj wszystko do float (nie numpy float32)
    total_length = float(sum(np.linalg.norm(e - s) * 1000 for s, e in lines_3d))
    
    # Oblicz dystans podróży (bez rysowania)
    travel_distance = 0.0
    for i in range(1, len(lines_3d)):
        prev_end = lines_3d[i-1][1]
        curr_start = lines_3d[i][0]
        travel_distance += float(np.linalg.norm(curr_start - prev_end) * 1000)
    
    drawing_time = total_length / Config.DRAWING_SPEED
    travel_time = travel_distance / Config.TRAVEL_SPEED
    
    stats = {
        "total_lines": int(len(lines_3d)),  # int zamiast numpy int
        "total_drawing_length_mm": round(float(total_length), 2),  # float zamiast numpy float32
        "total_travel_distance_mm": round(float(travel_distance), 2),
        "estimated_drawing_time_s": round(float(drawing_time), 2),
        "estimated_travel_time_s": round(float(travel_time), 2),
        "estimated_total_time_s": round(float(drawing_time + travel_time), 2),
        "estimated_total_time_min": round(float((drawing_time + travel_time) / 60), 2),
        "drawing_speed_mm_s": int(Config.DRAWING_SPEED),
        "travel_speed_mm_s": int(Config.TRAVEL_SPEED)
    }
    
    with open(os.path.join(output_dir, "robot_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Eksportowano dane robota:")
    print(f"  - {len(lines_3d)} linii")
    print(f"  - {total_length:.2f}mm rysowania")
    print(f"  - {travel_distance:.2f}mm podróży")
    print(f"  - ~{(drawing_time + travel_time)/60:.1f} min czasu")

# ---------------------
# Główna funkcja (zaktualizowana - lepsze wybieranie algorytmu)
# ---------------------
@measure_time
def main(point_cloud_path, camera_position=None, look_at_point=None, dev_mode=False):
    """Główna funkcja"""
    print("=" * 60)
    print("UPROSZCZONA WERSJA: CHMURA -> MESH -> ROBOT PATH")
    if NUMBA_AVAILABLE:
        print("🚀 NUMBA ACCELERATION ENABLED")
    print("=" * 60)
    
    # Wczytaj punkty
    points = load_point_cloud(point_cloud_path)
    print(f"Wczytano {len(points)} punktów")
    
    # Przetwórz
    clean_points = preprocess_point_cloud(points, Config.VOXEL_SIZE)
    
    # Zbuduj mesh (z szybszym depth)
    mesh = build_mesh(clean_points, Config.POISSON_DEPTH)
    
    # Oblicz centrum i wymiary
    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    max_extent = np.max(extent)
    
    print(f"Centrum meshu: {center}")
    print(f"Wymiary: {extent}")
    
    # Tryb deweloperski - test różnych pozycji kamery
    if dev_mode:
        print("\n" + "=" * 60)
        print("TRYB DEWELOPERSKI - Testowanie pozycji kamery")
        print("=" * 60)
        
        # Definicje pozycji testowych (względem centrum)
        test_positions = [
            # Nazwa, offset kamery, offset celu (opcjonalny)
            ("front_high", np.array([0, -max_extent*2.5, max_extent*1.5]), None),
            ("front_mid", np.array([0, -max_extent*2.5, max_extent*0.8]), None),
            ("front_low", np.array([0, -max_extent*2.5, max_extent*0.3]), None),
            
            ("side_right", np.array([max_extent*2.5, 0, max_extent*1.0]), None),
            ("side_left", np.array([-max_extent*2.5, 0, max_extent*1.0]), None),
            
            ("corner_high", np.array([max_extent*1.8, -max_extent*1.8, max_extent*1.8]), None),
            ("corner_low", np.array([max_extent*1.8, -max_extent*1.8, max_extent*0.5]), None),
            
            ("top", np.array([0, 0, max_extent*3.0]), None),
            
            ("angled_1", np.array([max_extent*1.5, -max_extent*2.0, max_extent*1.2]), None),
            ("angled_2", np.array([-max_extent*1.5, -max_extent*2.0, max_extent*1.2]), None),
        ]
        
        for name, cam_offset, target_offset in test_positions:
            print(f"\n>>> Testowanie pozycji: {name}")
            
            cam_pos = center + cam_offset
            look_at = center + target_offset if target_offset is not None else center
            
            print(f"    Kamera: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
            print(f"    Cel: [{look_at[0]:.2f}, {look_at[1]:.2f}, {look_at[2]:.2f}]")
            
            # Generuj linie
            lines_3d = generate_lines_from_mesh(mesh, cam_pos, Config.VISIBILITY_THRESHOLD)
            
            # Użyj szybszej optymalizacji
            try:
                optimized_lines = optimize_path_kdtree(lines_3d, max_lines=2000)
            except ImportError:
                optimized_lines = optimize_path(lines_3d, max_lines=2000)
            
            # Renderuj z nazwą zawierającą koordynaty
            filename = f"render_{name}_cam_{cam_pos[0]:.1f}_{cam_pos[1]:.1f}_{cam_pos[2]:.1f}_target_{look_at[0]:.1f}_{look_at[1]:.1f}_{look_at[2]:.1f}.png"
            output_path = os.path.join(Config.OUTPUT_DIR, filename)
            
            render_mesh(mesh, cam_pos, look_at, output_path, optimized_lines)
            
            # Eksportuj dane tylko dla pierwszej pozycji (aby nie spamować)
            if name == "front_high":
                export_robot_data(optimized_lines, Config.OUTPUT_DIR)
                export_hanwha_commands(optimized_lines, Config.OUTPUT_DIR)
        
        print("\n" + "=" * 60)
        print(f"Wygenerowano {len(test_positions)} testowych renderingów")
        print("Sprawdź folder output/ aby wybrać najlepszą pozycję")
        print("=" * 60)
        
    else:
        # Tryb normalny
        if camera_position is None:
            distance = max_extent * 2.5
            camera_position = center + np.array([
                0,
                -distance * 0.866,
                distance * 0.6
            ])
            look_at_point = center
            print(f"Auto-kamera: pozycja={camera_position}, cel={look_at_point}")
        
        print(f"Kamera: {camera_position}")
        print(f"Cel: {look_at_point}")
        
        # Generuj linie
        lines_3d = generate_lines_from_mesh(mesh, camera_position, Config.VISIBILITY_THRESHOLD)
        
        # Optymalizuj - wybierz najlepszy algorytm
        num_lines = len(lines_3d)
        
        if NUMBA_AVAILABLE:
            # Numba jest świetne dla małych/średnich zbiorów
            print(f"Używam optymalizacji Numba dla {num_lines} linii")
            optimized_lines = optimize_path(lines_3d, max_lines=2000)
        else:
            # Fallback do k-d tree (tylko jeśli scipy dostępne)
            try:
                from scipy.spatial import cKDTree
                print(f"Używam optymalizacji k-d tree dla {num_lines} linii")
                optimized_lines = optimize_path_kdtree(lines_3d, max_lines=2000)
            except ImportError:
                print(f"Używam standardowej optymalizacji dla {num_lines} linii")
                optimized_lines = optimize_path(lines_3d, max_lines=2000)
        
        # Renderuj z koordynatami w nazwie
        cam_str = f"cam_{camera_position[0]:.1f}_{camera_position[1]:.1f}_{camera_position[2]:.1f}"
        target_str = f"target_{look_at_point[0]:.1f}_{look_at_point[1]:.1f}_{look_at_point[2]:.1f}"
        filename = f"render_{cam_str}_{target_str}.png"
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        render_mesh(mesh, camera_position, look_at_point, 
                    os.path.join(Config.OUTPUT_DIR, filename), optimized_lines)
        
        # Eksportuj dane robota
        export_robot_data(optimized_lines, Config.OUTPUT_DIR)
        export_hanwha_commands(optimized_lines, Config.OUTPUT_DIR)
    
    # Zapisz mesh
    o3d.io.write_triangle_mesh(os.path.join(Config.OUTPUT_DIR, "mesh.ply"), mesh)
    print(f"Mesh zapisany: {Config.OUTPUT_DIR}/mesh.ply")
    
    print("\nGotowe!")

# ---------------------
# Entry point
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Konwersja chmury punktów na ścieżkę robota')
    parser.add_argument('point_cloud', help='Ścieżka do pliku chmury punktów (.ply, .pcd, .las, .laz)')
    parser.add_argument('--camera', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Pozycja kamery (X Y Z)')
    parser.add_argument('--target', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Punkt docelowy kamery (X Y Z)')
    parser.add_argument('--dev', action='store_true',
                       help='Tryb deweloperski - test różnych pozycji kamery')
    
    args = parser.parse_args()
    
    # Przygotuj parametry
    camera_position = None
    look_at_point = None
    
    if args.camera and args.target:
        camera_position = np.array(args.camera)
        look_at_point = np.array(args.target)
        print(f"Ustawiono kamerę: pozycja={camera_position}, cel={look_at_point}")
    elif args.camera or args.target:
        print("Błąd: Musisz podać zarówno --camera jak i --target")
        sys.exit(1)
    
    main(args.point_cloud, camera_position, look_at_point, dev_mode=args.dev)