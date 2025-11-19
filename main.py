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
    VOXEL_SIZE = 0.002
    POISSON_DEPTH = 6
    VISIBILITY_THRESHOLD = 0.1
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080
    DRAWING_SPEED = 100  # mm/s
    TRAVEL_SPEED = 150  # mm/s
    Z_LIFT = 5  # mm
    USE_PARALLEL = True
    MAX_WORKERS = min(4, cpu_count())
    MIN_TRIANGLES = 200
    MAX_TRIANGLES = 1000
    SMOOTHING_ITERATIONS = 2
    MAX_POINTS_FOR_ORIENT = 100000
    
    # NOWE: Parametry robota i kartki papieru
    PAPER_FORMAT = "A4"  # "A4" lub "A5"
    ROBOT_WORKSPACE_X = 300.0  # mm - maksymalny zakres X robota (przykład)
    ROBOT_WORKSPACE_Y = 300.0  # mm - maksymalny zakres Y robota
    ROBOT_WORKSPACE_Z = 100.0  # mm - maksymalny zakres Z robota
    ROBOT_SAFE_Z = 50.0        # mm - bezpieczna wysokość dla travel moves
    
    # Wymiary kartek (w mm)
    PAPER_SIZES = {
        "A4": (210.0, 297.0),   # Szerokość x Wysokość
        "A5": (148.0, 210.0),
        "A3": (297.0, 420.0),
        "A6": (105.0, 148.0)
    }
    
    # Marginesy na kartce (w mm)
    PAPER_MARGIN = 10.0  # odstęp od krawędzi kartki
    
    # Orientacja kartki
    PAPER_LANDSCAPE = False  # False = pionowo (portrait), True = poziomo (landscape)

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
# Loader chmur punktów i meshów (inteligentny)
# ---------------------
@measure_time
def load_point_cloud_or_mesh(path):
    """Inteligentnie wczytuje plik - wykrywa czy to mesh czy chmura punktów"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plik nie istnieje: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    print(f"Wczytywanie {path} (format: {ext})")
    
    # Najpierw próbuj wczytać jako mesh
    if ext in (".ply", ".obj", ".stl", ".off", ".gltf", ".glb"):
        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if len(mesh.triangles) > 0:
                print(f"✅ Wykryto MESH: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów")
                
                # Decymacja jeśli za duży
                if len(mesh.triangles) > Config.MAX_TRIANGLES:
                    print(f"⚠️ Mesh zbyt gęsty ({len(mesh.triangles)} trójkątów). Decymacja do {Config.MAX_TRIANGLES}...")
                    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=Config.MAX_TRIANGLES)
                    print(f"   Po decymacji: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów")
                
                return mesh, "mesh"
        except Exception as e:
            print(f"   Nie udało się wczytać jako mesh: {e}")
    
    # Jeśli nie mesh, wczytaj jako chmurę punktów
    print(f"   Wczytuję jako chmurę punktów...")
    
    if ext in (".las", ".laz"):
        import laspy
        las = laspy.open(path, mode='r')
        points = las.read()
        xyz = np.vstack((points.x, points.y, points.z)).T.astype(np.float32)
        print(f"✅ Wczytano chmurę punktów: {len(xyz)} punktów")
        return xyz, "pointcloud"
    else:
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            raise ValueError("Plik nie zawiera punktów ani meshu")
        points = np.asarray(pcd.points, dtype=np.float32)
        print(f"✅ Wczytano chmurę punktów: {len(points)} punktów")
        return points, "pointcloud"

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
# Renderowanie (NAPRAWIONE - z wizualizacją kartki)
# ---------------------
@measure_time
def render_mesh(mesh, camera_pos, look_at, output_path, lines_3d=None):
    """Renderuje mesh do obrazu z wizualizacją kartki i fallback do wireframe"""
    try:
        # Próba Open3D rendering
        os.environ['OPEN3D_CPU_RENDERING'] = 'true'
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=Config.IMAGE_WIDTH, height=Config.IMAGE_HEIGHT, visible=False)
        
        # ⭐ DODAJ: Wizualizację kartki papieru jako ramkę
        paper_width, paper_height = Config.PAPER_SIZES[Config.PAPER_FORMAT]
        if Config.PAPER_LANDSCAPE:
            paper_width, paper_height = paper_height, paper_width
        
        # Narysuj ramkę kartki na płaszczyźnie Z=ROBOT_SAFE_Z
        paper_z = Config.ROBOT_SAFE_Z
        paper_corners = np.array([
            [Config.PAPER_MARGIN, Config.PAPER_MARGIN, paper_z],
            [paper_width - Config.PAPER_MARGIN, Config.PAPER_MARGIN, paper_z],
            [paper_width - Config.PAPER_MARGIN, paper_height - Config.PAPER_MARGIN, paper_z],
            [Config.PAPER_MARGIN, paper_height - Config.PAPER_MARGIN, paper_z],
            [Config.PAPER_MARGIN, Config.PAPER_MARGIN, paper_z]  # Zamknij ramkę
        ])
        
        paper_frame = o3d.geometry.LineSet()
        paper_frame.points = o3d.utility.Vector3dVector(paper_corners)
        paper_frame.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(paper_corners)-1)])
        paper_frame.paint_uniform_color([0, 1, 0])  # Zielona ramka
        vis.add_geometry(paper_frame)
        
        # ⭐ DODAJ: Osie układu współrzędnych dla orientacji
        axis_length = max(paper_width, paper_height) * 0.3
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_length, 
            origin=[0, 0, paper_z]
        )
        vis.add_geometry(coord_frame)
        
        # Dodaj mesh
        mesh_copy = o3d.geometry.TriangleMesh(mesh)
        mesh_copy.compute_vertex_normals()
        mesh_copy.paint_uniform_color([0.7, 0.7, 0.7])
        vis.add_geometry(mesh_copy)
        
        # Dodaj linie robota
        if lines_3d:
            points = []
            line_indices = []
            for i, (start, end) in enumerate(lines_3d[:3000]):  # Limit
                points.extend([start, end])
                line_indices.append([i*2, i*2+1])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.paint_uniform_color([1, 0, 0])  # Czerwone linie
            vis.add_geometry(line_set)
        
        # Ustaw kamerę
        ctr = vis.get_view_control()
        if ctr is None:
            raise RuntimeError("View control failed")
        
        # ⭐ NAPRAWIONE: Ustaw parametry kamery w przestrzeni mm
        vis.poll_events()
        vis.update_renderer()
        
        # Oblicz wektor forward i up
        forward = (look_at - camera_pos)
        forward_len = np.linalg.norm(forward)
        if forward_len > 0:
            forward = forward / forward_len
        
        # Wybierz wektor up - unikaj równoległości z forward
        if abs(np.dot(forward, [0, 0, 1])) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 1, 0])
        
        # ⭐ WAŻNE: Ustaw zoom na podstawie odległości w mm
        ctr.set_lookat(look_at)
        ctr.set_front(forward)
        ctr.set_up(up)
        ctr.set_zoom(0.3)  # Dostosuj zoom dla większej skali mm
        
        vis.poll_events()
        vis.update_renderer()
        
        # Zapisz obraz
        image = vis.capture_screen_float_buffer(do_render=True)
        image_array = np.asarray(image)
        
        if image_array.size == 0:
            raise RuntimeError("Empty render buffer")
        
        image_bgr = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # ⭐ DODAJ: Naniesienie info o współrzędnych na obraz
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(image_bgr, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.putText(image_bgr, f"Camera: [{camera_pos[0]:.0f}, {camera_pos[1]:.0f}, {camera_pos[2]:.0f}] mm", 
                   (20, 35), font, 0.5, (255, 255, 255), 1)
        cv2.putText(image_bgr, f"Target: [{look_at[0]:.0f}, {look_at[1]:.0f}, {look_at[2]:.0f}] mm", 
                   (20, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(image_bgr, f"Distance: {forward_len:.0f} mm", 
                   (20, 85), font, 0.5, (255, 255, 255), 1)
        cv2.putText(image_bgr, f"Paper: {Config.PAPER_FORMAT}", 
                   (20, 110), font, 0.5, (0, 255, 0), 1)
        if lines_3d:
            cv2.putText(image_bgr, f"Lines: {len(lines_3d)}", 
                       (20, 135), font, 0.5, (0, 0, 255), 1)
        
        cv2.imwrite(output_path, image_bgr)
        print(f"Render 3D zapisany: {output_path}")
        
        vis.destroy_window()
        return output_path
        
    except Exception as e:
        print(f"Błąd renderowania 3D: {e}")
        import traceback
        traceback.print_exc()
        print("Przełączam na wireframe 2D...")
        return render_wireframe_2d(mesh, camera_pos, look_at, output_path, lines_3d)

def render_wireframe_2d(mesh, camera_pos, look_at, output_path, lines_3d=None):
    """Fallback: renderowanie wireframe 2D - NAPRAWIONE dla mm"""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Tło z gradientem
    img = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3), dtype=np.uint8)
    for y in range(Config.IMAGE_HEIGHT):
        gray = int(30 + (60 - 30) * (y / Config.IMAGE_HEIGHT))
        img[y, :] = [gray, gray, gray]
    
    # ⭐ NAPRAWIONE: Projekcja uwzględniająca skalę mm
    view_dir = (look_at - camera_pos)
    view_len = np.linalg.norm(view_dir)
    if view_len > 0:
        view_dir = view_dir / view_len
    
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
        cv2.putText(img, "No valid vertices", (50, Config.IMAGE_HEIGHT//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imwrite(output_path, img)
        return output_path
    
    # ⭐ NAPRAWIONE: Skalowanie uwzględniające rzeczywiste wymiary w mm
    min_vals = np.min(proj_vertices_clean, axis=0)
    max_vals = np.max(proj_vertices_clean, axis=0)
    ranges = max_vals - min_vals
    ranges = np.where(ranges < 1e-3, 1.0, ranges)  # Zmieniono próg z 1e-6 na 1e-3 (dla mm)
    
    margin = 0.1
    scale = min(Config.IMAGE_WIDTH * (1-2*margin) / ranges[0], 
                Config.IMAGE_HEIGHT * (1-2*margin) / ranges[1])
    
    center = (min_vals + max_vals) / 2
    scaled_vertices = (proj_vertices - center) * scale
    scaled_vertices[:, 0] += Config.IMAGE_WIDTH // 2
    scaled_vertices[:, 1] += Config.IMAGE_HEIGHT // 2
    scaled_vertices = scaled_vertices.astype(int)
    
    # ⭐ DODAJ: Rysuj ramkę kartki na projekcji
    paper_width, paper_height = Config.PAPER_SIZES[Config.PAPER_FORMAT]
    if Config.PAPER_LANDSCAPE:
        paper_width, paper_height = paper_height, paper_width
    
    # Rogi kartki w przestrzeni 3D
    paper_corners_3d = np.array([
        [Config.PAPER_MARGIN, Config.PAPER_MARGIN, Config.ROBOT_SAFE_Z],
        [paper_width - Config.PAPER_MARGIN, Config.PAPER_MARGIN, Config.ROBOT_SAFE_Z],
        [paper_width - Config.PAPER_MARGIN, paper_height - Config.PAPER_MARGIN, Config.ROBOT_SAFE_Z],
        [Config.PAPER_MARGIN, paper_height - Config.PAPER_MARGIN, Config.ROBOT_SAFE_Z],
    ])
    
    # Projektuj ramkę
    if abs(view_dir[2]) > 0.7:
        paper_corners_2d = paper_corners_3d[:, :2]
    elif abs(view_dir[1]) > 0.7:
        paper_corners_2d = paper_corners_3d[:, [0, 2]]
    else:
        paper_corners_2d = paper_corners_3d[:, [1, 2]]
    
    paper_scaled = (paper_corners_2d - center) * scale
    paper_scaled[:, 0] += Config.IMAGE_WIDTH // 2
    paper_scaled[:, 1] += Config.IMAGE_HEIGHT // 2
    paper_scaled = paper_scaled.astype(int)
    
    # Rysuj ramkę kartki
    cv2.polylines(img, [paper_scaled], True, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Rysuj trójkąty wireframe
    triangle_count = 0
    for tri in triangles:
        if all(valid_mask[idx] for idx in tri):
            pts = scaled_vertices[tri]
            if all(0 <= p[0] < Config.IMAGE_WIDTH and 0 <= p[1] < Config.IMAGE_HEIGHT for p in pts):
                cv2.polylines(img, [pts], True, (180, 180, 180), 1, cv2.LINE_AA)
                triangle_count += 1
    
    # Rysuj linie robota
    line_count = 0
    if lines_3d:
        for start_3d, end_3d in lines_3d[:5000]:
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
    
    # Panel informacyjny - rozszerzony
    cv2.rectangle(img, (0, 0), (500, 200), (20, 20, 20), -1)
    cv2.putText(img, f"Wireframe 2D ({axis_labels[0]}-{axis_labels[1]})", (15, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Vertices: {len(vertices):,}", (15, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"Triangles: {triangle_count:,}/{len(triangles):,}", (15, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if lines_3d:
        cv2.putText(img, f"Robot lines: {line_count:,}/{len(lines_3d):,}", (15, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Info o kamerze
    cv2.putText(img, f"Camera: [{camera_pos[0]:.0f}, {camera_pos[1]:.0f}, {camera_pos[2]:.0f}] mm", 
               (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(img, f"Target: [{look_at[0]:.0f}, {look_at[1]:.0f}, {look_at[2]:.0f}] mm", 
               (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(img, f"Paper: {Config.PAPER_FORMAT} (green frame)", 
               (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    print(f"Narysowano {triangle_count} trójkątów wireframe + {line_count} linii robota")
    cv2.imwrite(output_path, img)
    print(f"Wireframe 2D zapisany: {output_path}")
    return output_path

# ---------------------
# Generowanie SVG z siatki
# ---------------------
@measure_time
def generate_svg_from_mesh(mesh, camera_pos, look_at, output_path, lines_3d=None):
    """
    Generuje plik SVG z siatki - identyczny jak PNG, ale bez statystyk.
    Linie są czarne, tło przezroczyste, rozmiar jak kartka papieru.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Pobierz rozmiar kartki w mm
    paper_width, paper_height = Config.PAPER_SIZES[Config.PAPER_FORMAT]
    if Config.PAPER_LANDSCAPE:
        paper_width, paper_height = paper_height, paper_width
    
    # Konwersja mm na punkty (1mm = 3.7795275591 punktów, dla 96 DPI)
    # Ale SVG używa jednostek użytkownika, więc zostaniemy przy mm
    svg_width = paper_width
    svg_height = paper_height
    
    # Projekcja - identyczna jak w render_wireframe_2d
    view_dir = (look_at - camera_pos)
    view_len = np.linalg.norm(view_dir)
    if view_len > 0:
        view_dir = view_dir / view_len
    
    # Wybierz osie projekcji
    if abs(view_dir[2]) > 0.7:  # Z góry/dołu
        proj_vertices = vertices[:, :2]
    elif abs(view_dir[1]) > 0.7:  # Z przodu/tyłu
        proj_vertices = vertices[:, [0, 2]]
    else:  # Z boku
        proj_vertices = vertices[:, [1, 2]]
    
    # Usuń NaN/Inf
    valid_mask = np.isfinite(proj_vertices).all(axis=1)
    proj_vertices_clean = proj_vertices[valid_mask]
    
    if len(proj_vertices_clean) == 0:
        print("Brak prawidłowych wierzchołków dla SVG")
        return None
    
    # Skalowanie - dopasuj do rozmiaru kartki z marginesem
    min_vals = np.min(proj_vertices_clean, axis=0)
    max_vals = np.max(proj_vertices_clean, axis=0)
    ranges = max_vals - min_vals
    ranges = np.where(ranges < 1e-3, 1.0, ranges)
    
    margin_mm = Config.PAPER_MARGIN
    scale = min((svg_width - 2*margin_mm) / ranges[0], 
                (svg_height - 2*margin_mm) / ranges[1])
    
    center = (min_vals + max_vals) / 2
    scaled_vertices = (proj_vertices - center) * scale
    scaled_vertices[:, 0] += svg_width / 2
    scaled_vertices[:, 1] += svg_height / 2
    
    # Rogi kartki w przestrzeni 3D (dla opcjonalnej ramki)
    paper_corners_3d = np.array([
        [Config.PAPER_MARGIN, Config.PAPER_MARGIN, Config.ROBOT_SAFE_Z],
        [paper_width - Config.PAPER_MARGIN, Config.PAPER_MARGIN, Config.ROBOT_SAFE_Z],
        [paper_width - Config.PAPER_MARGIN, paper_height - Config.PAPER_MARGIN, Config.ROBOT_SAFE_Z],
        [Config.PAPER_MARGIN, paper_height - Config.PAPER_MARGIN, Config.ROBOT_SAFE_Z],
    ])
    
    # Projektuj ramkę
    if abs(view_dir[2]) > 0.7:
        paper_corners_2d = paper_corners_3d[:, :2]
    elif abs(view_dir[1]) > 0.7:
        paper_corners_2d = paper_corners_3d[:, [0, 2]]
    else:
        paper_corners_2d = paper_corners_3d[:, [1, 2]]
    
    paper_scaled = (paper_corners_2d - center) * scale
    paper_scaled[:, 0] += svg_width / 2
    paper_scaled[:, 1] += svg_height / 2
    
    # Rozpocznij generowanie SVG
    svg_lines = []
    svg_lines.append(f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg_lines.append(f'<svg width="{svg_width}mm" height="{svg_height}mm" ')
    svg_lines.append(f'     viewBox="0 0 {svg_width} {svg_height}" ')
    svg_lines.append(f'     xmlns="http://www.w3.org/2000/svg" version="1.1">')
    svg_lines.append(f'  <title>Mesh Wireframe - {Config.PAPER_FORMAT}</title>')
    svg_lines.append(f'  <desc>Generated from point cloud mesh</desc>')
    svg_lines.append(f'  ')
    
    # Grupa dla trójkątów wireframe
    svg_lines.append(f'  <g id="triangles" stroke="black" stroke-width="0.5" fill="none" stroke-linecap="round" stroke-linejoin="round">')
    
    triangle_count = 0
    for tri in triangles:
        if all(valid_mask[idx] for idx in tri):
            pts = scaled_vertices[tri]
            if all(0 <= p[0] <= svg_width and 0 <= p[1] <= svg_height for p in pts):
                # Rysuj trójkąt jako polygon
                points_str = " ".join([f"{p[0]:.3f},{p[1]:.3f}" for p in pts])
                svg_lines.append(f'    <polygon points="{points_str}" />')
                triangle_count += 1
    
    svg_lines.append(f'  </g>')
    
    # Grupa dla linii robota (jeśli są)
    if lines_3d:
        svg_lines.append(f'  ')
        svg_lines.append(f'  <g id="robot_lines" stroke="black" stroke-width="1.0" fill="none" stroke-linecap="round">')
        
        line_count = 0
        for start_3d, end_3d in lines_3d[:5000]:
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
            start_scaled = (start_2d - center) * scale + [svg_width/2, svg_height/2]
            end_scaled = (end_2d - center) * scale + [svg_width/2, svg_height/2]
            
            if (0 <= start_scaled[0] <= svg_width and 0 <= start_scaled[1] <= svg_height and
                0 <= end_scaled[0] <= svg_width and 0 <= end_scaled[1] <= svg_height):
                svg_lines.append(f'    <line x1="{start_scaled[0]:.3f}" y1="{start_scaled[1]:.3f}" '
                               f'x2="{end_scaled[0]:.3f}" y2="{end_scaled[1]:.3f}" />')
                line_count += 1
        
        svg_lines.append(f'  </g>')
        print(f"Narysowano {line_count} linii robota w SVG")
    
    svg_lines.append(f'</svg>')
    
    # Zapisz do pliku
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_lines))
    
    print(f"SVG zapisany: {output_path}")
    print(f"  • Rozmiar: {svg_width}mm x {svg_height}mm ({Config.PAPER_FORMAT})")
    print(f"  • Trójkąty: {triangle_count}")
    return output_path

# ---------------------
# Eksport dla robota Hanwha HCR-3A (NAPRAWIONY - bez podwójnego mnożenia)
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
        f.write(f"# Liczba linii: {len(lines_3d)}\n")
        f.write(f"# UWAGA: Współrzędne już w milimetrach (po skalowaniu)\n\n")
        
        # Inicjalizacja
        f.write("ROBOT_INIT\n")
        f.write(f"SET_SPEED DRAW={Config.DRAWING_SPEED} TRAVEL={Config.TRAVEL_SPEED}\n")
        f.write(f"SET_Z_LIFT {Config.Z_LIFT}\n")
        f.write("TOOL_ON\n\n")
        
        # Pierwsza pozycja
        first_start = lines_3d[0][0]
        f.write("# Przejście do pozycji startowej\n")
        f.write(f"MOVE_SAFE {first_start[0]:.3f} {first_start[1]:.3f} {first_start[2]:.3f}\n")  # Bez *1000!
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
                    f.write(f"MOVE_FAST {start[0]:.3f} {start[1]:.3f} {start[2]:.3f}\n")  # Bez *1000!
                    f.write("PEN_DOWN\n")
            
            # Rysuj linię
            f.write(f"DRAW_LINE {end[0]:.3f} {end[1]:.3f} {end[2]:.3f}\n")  # Bez *1000!
        
        # Zakończenie
        f.write("\n# Zakończenie\n")
        f.write("PEN_UP\n")
        f.write("MOVE_HOME\n")
        f.write("TOOL_OFF\n")
        f.write("ROBOT_END\n")
    
    print(f"Komendy Hanwha zapisane: {commands_file}")
    return commands_file

# ---------------------
# Eksport danych robota (NAPRAWIONY - współrzędne już w mm)
# ---------------------
@measure_time
def export_robot_data(lines_3d, output_dir):
    """Eksportuje dane dla robota - współrzędne już w mm po skalowaniu"""
    if not lines_3d:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Punkty (JUŻ W MILIMETRACH - nie mnóż przez 1000!)
    points = np.vstack([[s, e] for s, e in lines_3d])
    np.savetxt(os.path.join(output_dir, "robot_points_mm.txt"), points, fmt="%.3f",
               header="Współrzędne w mm (po skalowaniu do kartki)\nFormat: x y z")
    
    # Linie (JUŻ W MILIMETRACH)
    with open(os.path.join(output_dir, "robot_lines_mm.txt"), 'w') as f:
        f.write("# Format: start_x start_y start_z end_x end_y end_z (wszystko w mm)\n")
        f.write("# UWAGA: Współrzędne już przeskalowane do kartki!\n")
        for s, e in lines_3d:
            f.write(f"{s[0]:.3f} {s[1]:.3f} {s[2]:.3f} {e[0]:.3f} {e[1]:.3f} {e[2]:.3f}\n")
    
    # Statystyki - współrzędne już w mm, więc NIE mnóż przez 1000
    total_length = float(sum(np.linalg.norm(e - s) for s, e in lines_3d))
    
    # Oblicz dystans podróży (bez rysowania)
    travel_distance = 0.0
    for i in range(1, len(lines_3d)):
        prev_end = lines_3d[i-1][1]
        curr_start = lines_3d[i][0]
        travel_distance += float(np.linalg.norm(curr_start - prev_end))
    
    drawing_time = total_length / Config.DRAWING_SPEED
    travel_time = travel_distance / Config.TRAVEL_SPEED
    
    # Sprawdź zakresy współrzędnych
    all_points = np.vstack([np.vstack([s, e]) for s, e in lines_3d])
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    
    stats = {
        "total_lines": int(len(lines_3d)),
        "total_drawing_length_mm": round(float(total_length), 2),
        "total_travel_distance_mm": round(float(travel_distance), 2),
        "estimated_drawing_time_s": round(float(drawing_time), 2),
        "estimated_travel_time_s": round(float(travel_time), 2),
        "estimated_total_time_s": round(float(drawing_time + travel_time), 2),
        "estimated_total_time_min": round(float((drawing_time + travel_time) / 60), 2),
        "drawing_speed_mm_s": int(Config.DRAWING_SPEED),
        "travel_speed_mm_s": int(Config.TRAVEL_SPEED),
        # Dodano: zakresy współrzędnych
        "coordinate_ranges_mm": {
            "x_min": round(float(min_coords[0]), 2),
            "x_max": round(float(max_coords[0]), 2),
            "y_min": round(float(min_coords[1]), 2),
            "y_max": round(float(max_coords[1]), 2),
            "z_min": round(float(min_coords[2]), 2),
            "z_max": round(float(max_coords[2]), 2)
        },
        "note": "Współrzędne już przeskalowane do kartki (w mm)"
    }
    
    with open(os.path.join(output_dir, "robot_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Eksportowano dane robota:")
    print(f"  - {len(lines_3d)} linii")
    print(f"  - {total_length:.2f}mm rysowania")
    print(f"  - {travel_distance:.2f}mm podróży")
    print(f"  - ~{(drawing_time + travel_time)/60:.1f} min czasu")
    print(f"\n  Zakresy współrzędnych (mm):")
    print(f"    X: [{min_coords[0]:.1f}, {max_coords[0]:.1f}]")
    print(f"    Y: [{min_coords[1]:.1f}, {max_coords[1]:.1f}]")
    print(f"    Z: [{min_coords[2]:.1f}, {max_coords[2]:.1f}]")

# ---------------------
# Funkcja skalująca mesh do wymiarów kartki (NAPRAWIONA - zachowuje proporcje)
# ---------------------
@measure_time
def scale_mesh_to_paper(mesh):
    """Skaluje i centruje mesh do wymiaru kartki papieru - ZACHOWUJE PROPORCJE"""
    
    # Pobierz wymiary kartki
    paper_width, paper_height = Config.PAPER_SIZES[Config.PAPER_FORMAT]
    
    # Orientacja
    if Config.PAPER_LANDSCAPE:
        paper_width, paper_height = paper_height, paper_width
        orientation = "landscape"
    else:
        orientation = "portrait"
    
    # Uwzględnij marginesy
    usable_width = paper_width - 2 * Config.PAPER_MARGIN
    usable_height = paper_height - 2 * Config.PAPER_MARGIN
    
    print(f"\n{'='*60}")
    print(f"SKALOWANIE DO KARTKI {Config.PAPER_FORMAT} ({orientation})")
    print(f"  • Rozmiar kartki: {paper_width:.1f} x {paper_height:.1f} mm")
    print(f"  • Marginesy: {Config.PAPER_MARGIN:.1f} mm")
    print(f"  • Obszar rysowania: {usable_width:.1f} x {usable_height:.1f} mm")
    print(f"{'='*60}\n")
    
    # Pobierz bbox oryginalnego meshu
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    center = bbox.get_center()
    
    print(f"Oryginalny mesh:")
    print(f"  • Wymiary: [{extent[0]:.6f} x {extent[1]:.6f} x {extent[2]:.6f}]")
    print(f"  • Centrum: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    print(f"  • Proporcje X:Y:Z = 1:{extent[1]/extent[0]:.3f}:{extent[2]/extent[0]:.3f}")
    
    # ⭐ KLUCZOWE: Oblicz JEDEN współczynnik skalowania dla WSZYSTKICH osi
    # aby zachować proporcje obiektu
    
    # Znajdź największy wymiar w płaszczyźnie XY
    max_xy_extent = max(extent[0], extent[1])
    
    # Oblicz skalę aby zmieścić w obszarze rysowania
    # (zachowując proporcje, skalujemy wszystko tak samo)
    scale_factor = min(usable_width / max_xy_extent, usable_height / max_xy_extent)
    
    # Sprawdź czy Z po przeskalowaniu nie będzie za wysokie
    scaled_z = extent[2] * scale_factor
    max_allowed_z = Config.ROBOT_WORKSPACE_Z - Config.ROBOT_SAFE_Z - 10.0
    
    if scaled_z > max_allowed_z:
        # Zmniejsz skalę jeśli Z byłoby za wysokie
        scale_factor = max_allowed_z / extent[2]
        print(f"  ⚠️  Dostosowano skalę ze względu na wysokość Z")
    
    print(f"\nWspółczynnik skalowania (jednolity dla X,Y,Z): {scale_factor:.6f}x")
    print(f"  • Po skalowaniu: [{extent[0]*scale_factor:.1f} x {extent[1]*scale_factor:.1f} x {extent[2]*scale_factor:.1f}] mm")
    
    # Zastosuj skalowanie
    vertices = np.asarray(mesh.vertices)
    
    # 1. Przesuń do początku układu współrzędnych (0,0,0)
    vertices = vertices - center
    
    # 2. Skaluj RÓWNOMIERNIE wszystkie osie (zachowaj proporcje!)
    vertices = vertices * scale_factor
    
    # 3. Oblicz centrum kartki
    offset_x = paper_width / 2.0
    offset_y = paper_height / 2.0
    
    # 4. Oblicz offset Z - dno obiektu ma być na ROBOT_SAFE_Z
    scaled_extent = extent * scale_factor
    offset_z = Config.ROBOT_SAFE_Z + scaled_extent[2] / 2.0  # Środek obiektu na SAFE_Z + połowa wysokości
    
    # 5. Przesuń do centrum kartki
    vertices[:, 0] += offset_x
    vertices[:, 1] += offset_y
    vertices[:, 2] += offset_z
    
    # 6. Upewnij się że dno jest na ROBOT_SAFE_Z
    min_z = np.min(vertices[:, 2])
    if min_z < Config.ROBOT_SAFE_Z:
        adjustment = Config.ROBOT_SAFE_Z - min_z
        vertices[:, 2] += adjustment
        print(f"  • Dostosowano Z o +{adjustment:.1f}mm aby dno było na ROBOT_SAFE_Z")
    
    # Zaktualizuj mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    # Nowy bbox
    new_bbox = mesh.get_axis_aligned_bounding_box()
    new_extent = new_bbox.get_extent()
    new_center = new_bbox.get_center()
    
    print(f"\nPrzeskalowany mesh:")
    print(f"  • Wymiary: [{new_extent[0]:.1f} x {new_extent[1]:.1f} x {new_extent[2]:.1f}] mm")
    print(f"  • Centrum: [{new_center[0]:.1f}, {new_center[1]:.1f}, {new_center[2]:.1f}] mm")
    print(f"  • Proporcje zachowane: X:Y:Z = 1:{new_extent[1]/new_extent[0]:.3f}:{new_extent[2]/new_extent[0]:.3f}")
    
    # Sprawdź czy mieści się w workspace robota
    min_pt = new_bbox.get_min_bound()
    max_pt = new_bbox.get_max_bound()
    
    print(f"\nZakres współrzędnych (mm):")
    print(f"  • X: [{min_pt[0]:.1f}, {max_pt[0]:.1f}] (workspace: 0 - {Config.ROBOT_WORKSPACE_X:.1f})")
    print(f"  • Y: [{min_pt[1]:.1f}, {max_pt[1]:.1f}] (workspace: 0 - {Config.ROBOT_WORKSPACE_Y:.1f})")
    print(f"  • Z: [{min_pt[2]:.1f}, {max_pt[2]:.1f}] (workspace: 0 - {Config.ROBOT_WORKSPACE_Z:.1f})")
    
    # Walidacja
    warnings = []
    if min_pt[0] < 0 or max_pt[0] > Config.ROBOT_WORKSPACE_X:
        warnings.append(f"❌ X poza workspace [{min_pt[0]:.1f}, {max_pt[0]:.1f}]")
    if min_pt[1] < 0 or max_pt[1] > Config.ROBOT_WORKSPACE_Y:
        warnings.append(f"❌ Y poza workspace [{min_pt[1]:.1f}, {max_pt[1]:.1f}]")
    if min_pt[2] < 0 or max_pt[2] > Config.ROBOT_WORKSPACE_Z:
        warnings.append(f"❌ Z poza workspace [{min_pt[2]:.1f}, {max_pt[2]:.1f}]")
    
    # Sprawdź czy mesh wypełnia rozsądnie kartkę
    fill_x = new_extent[0] / usable_width * 100
    fill_y = new_extent[1] / usable_height * 100
    print(f"\nWypełnienie kartki:")
    print(f"  • X: {fill_x:.1f}% obszaru")
    print(f"  • Y: {fill_y:.1f}% obszaru")
    
    if fill_x < 30 or fill_y < 30:
        print(f"  ⚠️  Obiekt zajmuje mało miejsca - można zwiększyć skalę")
    
    if warnings:
        print(f"\n⚠️  OSTRZEŻENIA:")
        for w in warnings:
            print(f"  {w}")
        print(f"  Zwiększ ROBOT_WORKSPACE_* lub zmień format kartki")
    else:
        print(f"\n✅ Mesh mieści się w workspace robota!")
    
    # Zapisz informacje o skalowaniu
    scale_info = {
        "paper_format": Config.PAPER_FORMAT,
        "paper_orientation": orientation,
        "paper_size_mm": [float(paper_width), float(paper_height)],
        "usable_area_mm": [float(usable_width), float(usable_height)],
        "scale_factor": float(scale_factor),
        "original_dimensions": [float(extent[0]), float(extent[1]), float(extent[2])],
        "mesh_dimensions_mm": [float(new_extent[0]), float(new_extent[1]), float(new_extent[2])],
        "mesh_center_mm": [float(new_center[0]), float(new_center[1]), float(new_center[2])],
        "bounds_min_mm": [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
        "bounds_max_mm": [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])],
        "fits_in_workspace": len(warnings) == 0,
        "fill_percentage_x": float(fill_x),
        "fill_percentage_y": float(fill_y)
    }
        
    # Upewnij się że katalog istnieje
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    scale_info_path = os.path.join(Config.OUTPUT_DIR, "scaling_info.json")
    with open(scale_info_path, 'w') as f:
        json.dump(scale_info, f, indent=2)
    
    print(f"\nInfo o skalowaniu zapisano: {scale_info_path}")
    print(f"{'='*60}\n")
    
    return mesh

# ---------------------
# Główna funkcja (zaktualizowana - NAPRAWIONA kamera po skalowaniu)
# ---------------------
@measure_time
def main(point_cloud_path, camera_position=None, look_at_point=None, dev_mode=False):
    """Główna funkcja"""
    print("=" * 60)
    print("INTELIGENTNA WERSJA: AUTO-DETECT MESH/POINTCLOUD")
    if NUMBA_AVAILABLE:
        print("🚀 NUMBA ACCELERATION ENABLED")
    print("=" * 60)
    
    # Inteligentne wczytywanie - wykryj czy mesh czy chmura
    data, data_type = load_point_cloud_or_mesh(point_cloud_path)
    
    if data_type == "mesh":
        # Gotowy mesh - użyj bezpośrednio!
        print(f"\n{'='*60}")
        print("🎯 Wykryto gotowy mesh - pomijam preprocessing!")
        print(f"{'='*60}\n")
        mesh = data
        
        # Opcjonalnie: delikatna walidacja i czyszczenie
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        
        # Upewnij się że są normalne
        if len(mesh.triangle_normals) == 0:
            mesh.compute_triangle_normals()
        if len(mesh.vertex_normals) == 0:
            mesh.compute_vertex_normals()
        
        print(f"Mesh po czyszczeniu: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów\n")
    else:
        # Chmura punktów - pełny pipeline
        print(f"\n{'='*60}")
        print("☁️  Chmura punktów - pełny pipeline preprocessing")
        print(f"{'='*60}\n")
        points = data
        print(f"Wczytano {len(points)} punktów")
        
        # Przetwórz
        clean_points = preprocess_point_cloud(points, Config.VOXEL_SIZE)
        
        # Zbuduj mesh
        mesh = build_mesh(clean_points, Config.POISSON_DEPTH)
    
    # ⭐ SKALUJ DO KARTKI PAPIERU ⭐
    mesh = scale_mesh_to_paper(mesh)
    
    # ⭐ WAŻNE: Po skalowaniu oblicz NOWE centrum i wymiary (w mm)
    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    
    # ⭐ NAPRAWIONE: max_extent to teraz najdłuższy wymiar w mm (nie jednostkach oryginalnych)
    max_extent = np.max(extent)
    
    print(f"\nPo skalowaniu do kartki:")
    print(f"  • Centrum meshu: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] mm")
    print(f"  • Wymiary: [{extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f}] mm")
    print(f"  • Maksymalny wymiar: {max_extent:.1f} mm\n")
    
    # Tryb deweloperski
    if dev_mode:
        print("\n" + "=" * 60)
        print("TRYB DEWELOPERSKI - Testowanie pozycji kamery")
        print("=" * 60)
        
        # ⭐ NAPRAWIONE: Pozycje testowe używają rzeczywistych wymiarów w mm
        # Dla kartki A4 (~210x297mm) ustaw kamery na sensownych odległościach
        
        # Oblicz bezpieczną odległość kamery (2-3x większy wymiar kartki)
        safe_distance = max(extent[0], extent[1]) * 2.5  # ~500-750mm dla A4
        
        test_positions = [
            # Z przodu - różne wysokości
            ("front_high", np.array([0, -safe_distance, max_extent * 2.0]), None),
            ("front_mid", np.array([0, -safe_distance, max_extent]), None),
            ("front_low", np.array([0, -safe_distance, max_extent * 0.5]), None),
            
            # Z boków
            ("side_right", np.array([safe_distance, 0, max_extent]), None),
            ("side_left", np.array([-safe_distance, 0, max_extent]), None),
            
            # Z rogów - bardziej artystyczne
            ("corner_high", np.array([safe_distance * 0.7, -safe_distance * 0.7, max_extent * 1.5]), None),
            ("corner_low", np.array([safe_distance * 0.7, -safe_distance * 0.7, max_extent * 0.3]), None),
            
            # Z góry (widok płaski)
            ("top", np.array([0, 0, safe_distance]), None),
            
            # Kąty pod nachyleniem
            ("angled_1", np.array([safe_distance * 0.6, -safe_distance * 0.8, max_extent * 1.2]), None),
            ("angled_2", np.array([-safe_distance * 0.6, -safe_distance * 0.8, max_extent * 1.2]), None),
        ]
        
        print(f"\nBezpieczna odległość kamery: {safe_distance:.1f} mm\n")
        
        for idx, (name, cam_offset, target_offset) in enumerate(test_positions, 1):
            print(f"[{idx}/{len(test_positions)}] Testowanie pozycji: {name}")
            
            cam_pos = center + cam_offset
            look_at = center + target_offset if target_offset is not None else center
            
            print(f"     Kamera: [{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}] mm")
            print(f"     Cel: [{look_at[0]:.1f}, {look_at[1]:.1f}, {look_at[2]:.1f}] mm")
            print(f"     Odległość: {np.linalg.norm(cam_pos - look_at):.1f} mm")
            
            # ⭐ ROTACJA: Tylko dla widoku "top" - obrót o 180 stopni
            mesh_to_use = mesh
            if name == "top":
                print(f"     🔄 Obracam mesh o 180° dla widoku z góry...")
                mesh_to_use = o3d.geometry.TriangleMesh(mesh)  # Kopia
                R = mesh_to_use.get_rotation_matrix_from_xyz((0, 0, np.pi))
                mesh_to_use.rotate(R, center=center)
            
            # Generuj linie
            lines_3d = generate_lines_from_mesh(mesh_to_use, cam_pos, Config.VISIBILITY_THRESHOLD)
            
            # Użyj szybszej optymalizacji
            try:
                optimized_lines = optimize_path_kdtree(lines_3d, max_lines=2000)
            except ImportError:
                optimized_lines = optimize_path(lines_3d, max_lines=2000)
            
            # Renderuj PNG
            filename = f"render_{name}_cam_{cam_pos[0]:.1f}_{cam_pos[1]:.1f}_{cam_pos[2]:.1f}_target_{look_at[0]:.1f}_{look_at[1]:.1f}_{look_at[2]:.1f}.png"
            output_path = os.path.join(Config.OUTPUT_DIR, filename)
            
            render_mesh(mesh_to_use, cam_pos, look_at, output_path, optimized_lines)
            
            # Generuj SVG (bez statystyk, identyczny jak PNG)
            svg_filename = filename.replace('.png', '.svg')
            svg_output_path = os.path.join(Config.OUTPUT_DIR, svg_filename)
            generate_svg_from_mesh(mesh_to_use, cam_pos, look_at, svg_output_path, optimized_lines)
            
            # Eksportuj dane tylko dla pierwszej pozycji
            if name == "front_high":
                export_robot_data(optimized_lines, Config.OUTPUT_DIR)
                export_hanwha_commands(optimized_lines, Config.OUTPUT_DIR)
            
            print("")
        
        print("\n" + "=" * 60)
        print(f"✅ Wygenerowano {len(test_positions)} testowych renderingów")
        print(f"   Sprawdź folder: {Config.OUTPUT_DIR}/")
        print("=" * 60)
        
    else:
        # Tryb normalny
        if camera_position is None:
            # ⭐ DOMYŚLNY WIDOK: TOP (z góry) z rotacją 180°
            safe_distance = max(extent[0], extent[1]) * 2.5  # ~500-750mm dla A4
            
            camera_position = center + np.array([
                0,                          # Centralne X
                0,                          # Centralne Y
                safe_distance               # Z góry
            ])
            look_at_point = center
            
            print(f"\n🔄 Auto-kamera: TOP (widok z góry) z rotacją 180°")
            print(f"  • Pozycja: [{camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f}]")
            print(f"  • Cel: [{look_at_point[0]:.1f}, {look_at_point[1]:.1f}, {look_at_point[2]:.1f}]")
            print(f"  • Odległość: {np.linalg.norm(camera_position - look_at_point):.1f} mm\n")
            
            # Rotacja meshu o 180° dla widoku z góry
            print(f"  🔄 Obracam mesh o 180° dla widoku z góry...")
            mesh_rotated = o3d.geometry.TriangleMesh(mesh)  # Kopia
            R = mesh_rotated.get_rotation_matrix_from_xyz((0, 0, np.pi))
            mesh_rotated.rotate(R, center=center)
            mesh = mesh_rotated
        else:
            print(f"\nUżywam podanej kamery:")
            print(f"  • Pozycja: [{camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f}]")
            print(f"  • Cel: [{look_at_point[0]:.1f}, {look_at_point[1]:.1f}, {look_at_point[2]:.1f}]")
            print(f"  • Odległość: {np.linalg.norm(camera_position - look_at_point):.1f} mm\n")
        
        # Generuj linie
        lines_3d = generate_lines_from_mesh(mesh, camera_position, Config.VISIBILITY_THRESHOLD)
        
        # Optymalizuj
        num_lines = len(lines_3d)
        
        if NUMBA_AVAILABLE:
            print(f"Używam optymalizacji Numba dla {num_lines} linii")
            optimized_lines = optimize_path(lines_3d, max_lines=2000)
        else:
            try:
                from scipy.spatial import cKDTree
                print(f"Używam optymalizacji k-d tree dla {num_lines} linii")
                optimized_lines = optimize_path_kdtree(lines_3d, max_lines=2000)
            except ImportError:
                print(f"Używam standardowej optymalizacji dla {num_lines} linii")
                optimized_lines = optimize_path(lines_3d, max_lines=2000)
        
        # Renderuj PNG
        filename = f"render_output_cam_{camera_position[0]:.1f}_{camera_position[1]:.1f}_{camera_position[2]:.1f}_target_{look_at_point[0]:.1f}_{look_at_point[1]:.1f}_{look_at_point[2]:.1f}.png"
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        render_mesh(mesh, camera_position, look_at_point, 
                    os.path.join(Config.OUTPUT_DIR, filename), optimized_lines)
        
        # Generuj SVG (bez statystyk, identyczny jak PNG)
        svg_filename = filename.replace('.png', '.svg')
        svg_output_path = os.path.join(Config.OUTPUT_DIR, svg_filename)
        generate_svg_from_mesh(mesh, camera_position, look_at_point, svg_output_path, optimized_lines)
        
        # Eksportuj dane robota
        export_robot_data(optimized_lines, Config.OUTPUT_DIR)
        export_hanwha_commands(optimized_lines, Config.OUTPUT_DIR)
    
    # Zapisz mesh
    mesh_path = os.path.join(Config.OUTPUT_DIR, "mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"\nMesh zapisany: {mesh_path}")
    
    print("\n" + "="*60)
    print("✅ WSZYSTKO GOTOWE!")
    print(f"   Pliki wyjściowe: {Config.OUTPUT_DIR}/")
    print("="*60 + "\n")

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