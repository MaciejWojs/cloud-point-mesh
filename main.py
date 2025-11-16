#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import open3d as o3d
import time
from functools import wraps, lru_cache
import os
from scipy.spatial import cKDTree
import json

# ---------------------
# Konfiguracja
# ---------------------
class Config:
    OUTPUT_DIR = "./output"
    VOXEL_SIZE = 0.02
    POISSON_DEPTH = 10
    SIMPLIFICATION_RATIO = 0.25
    VISIBILITY_THRESHOLD = 0.1
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080
    DRAWING_SPEED = 50  # mm/s
    TRAVEL_SPEED = 100  # mm/s
    Z_LIFT = 5  # mm

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
# Loader chmur punktów
# ---------------------
@lru_cache(maxsize=8)
@measure_time
def load_point_cloud(path):
    """Wczytuje chmurę punktów z różnych formatów"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plik nie istnieje: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    print(f"Wczytywanie {path} (format: {ext})")
    
    try:
        if ext in (".las", ".laz"):
            return load_las_file(path)
        elif ext in [".ply", ".pcd", ".xyz", ".obj", ".off", ".stl"]:
            return load_3d_file(path)
        elif ext in [".csv", ".txt"]:
            return load_text_file(path)
        else:
            raise ValueError(f"Nieobsługiwany format pliku: {ext}")
    except Exception as e:
        print(f"Błąd wczytywania {path}: {e}")
        raise

def load_las_file(path):
    """Wczytuje plik LAS/LAZ"""
    import laspy
    las = laspy.open(path, mode='r')
    points = las.read()
    xyz = np.vstack((points.x, points.y, points.z)).T.astype(np.float32)
    
    # Normalizacja współrzędnych jeśli są zbyt duże
    if np.max(np.abs(xyz)) > 1000:
        print("Normalizacja współrzędnych LAS...")
        xyz = xyz - np.mean(xyz, axis=0)
    
    # Intensywność lub kolory
    if hasattr(points, "intensity"):
        intensity = np.asarray(points.intensity, dtype=np.float32)
        intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    else:
        intensity = np.ones(len(xyz), dtype=np.float32)
    
    return np.hstack([xyz, intensity[:, None]])

def load_3d_file(path):
    """Wczytuje pliki 3D (PLY, PCD, OBJ, etc.)"""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".stl":
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_poisson_disk(100000)
    else:
        pcd = o3d.io.read_point_cloud(path)
    
    if not pcd.has_points() or len(pcd.points) == 0:
        raise ValueError("Plik nie zawiera punktów")
    
    points = np.asarray(pcd.points, dtype=np.float32)
    
    # Normalizacja jeśli współrzędne są zbyt duże
    if np.max(np.abs(points)) > 1000:
        print("Normalizacja współrzędnych 3D...")
        points = points - np.mean(points, axis=0)
    
    # Dodanie intensywności z kolorów lub stałej wartości
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
        intensity = colors.mean(axis=1, keepdims=True)
    else:
        intensity = np.ones((len(points), 1), dtype=np.float32)
    
    return np.hstack([points, intensity])

def load_text_file(path):
    """Wczytuje pliki tekstowe (CSV, TXT)"""
    data = np.loadtxt(path, delimiter=None, dtype=np.float32)
    
    if data.ndim != 2:
        raise ValueError("Nieprawidłowy format danych")
    
    # Automatyczne wykrywanie kolumn
    if data.shape[1] >= 3:
        points = data[:, :3]
        if data.shape[1] >= 4:
            intensity = data[:, 3:4]
        else:
            intensity = np.ones((len(points), 1), dtype=np.float32)
    else:
        raise ValueError("Plik musi zawierać co najmniej 3 kolumny (X,Y,Z)")
    
    return np.hstack([points, intensity])

# ---------------------
# Przetwarzanie wstępne chmury punktów
# ---------------------
@measure_time
def preprocess_point_cloud(points, voxel_size=0.02):
    """Czyszczenie i downsampling chmury punktów"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Usuwanie outlierów statystycznych
    pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Usunięto {len(points) - len(inliers)} outlierów")
    
    # Downsampling
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"Po downsamplingu: {len(pcd.points)} punktów")
    
    return np.asarray(pcd.points, dtype=np.float32)

# ---------------------
# Budowa meshu
# ---------------------
@measure_time
def build_mesh(points, method="poisson", depth=10, simplify=True):
    """Buduje siatkę trójkątną z chmury punktów"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estymacja normalnych
    print("Estymacja normalnych...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    
    if method == "ball_pivoting":
        # Ball Pivoting
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    else:
        # Poisson Reconstruction
        print("Rekonstrukcja Poisson...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=True)
        
        # Usuwanie wierzchołków o niskiej gęstości
        if len(densities) > 0:
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.05)
            vertices_to_keep = densities > density_threshold
            mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    
    # Wygładzanie
    print("Wygładzanie meshu...")
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=3)
    
    # Uproszczenie
    if simplify and len(mesh.triangles) > 1000:
        target_triangles = max(1000, int(len(mesh.triangles) * Config.SIMPLIFICATION_RATIO))
        print(f"Upraszczanie meshu do {target_triangles} trójkątów...")
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    
    # Oblicz normalne
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    print(f"Utworzono mesh: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów")
    return mesh

# ---------------------
# Generowanie linii 3D z meshu
# ---------------------
@measure_time
def generate_simplified_lines_from_mesh(mesh, camera_position, threshold=0.1):
    """Generuje uproszczone linie 3D z widocznych krawędzi meshu"""
    if len(mesh.triangles) == 0:
        print("Mesh nie zawiera trójkątów")
        return []

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Oblicz normalne trójkątów jeśli nie są dostępne
    if len(mesh.triangle_normals) == 0:
        mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    
    # Środki trójkątów i wektory do kamery
    centers = np.mean(vertices[triangles], axis=1)
    view_dirs = camera_position - centers
    view_dirs_norm = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)
    
    # Test widoczności (backface culling)
    dot_products = np.einsum('ij,ij->i', normals, view_dirs_norm)
    visible_triangles = triangles[dot_products > threshold]
    
    if len(visible_triangles) == 0:
        print("Brak widocznych trójkątów")
        return []
    
    # Zbierz wszystkie krawędzie widocznych trójkątów
    edges_set = set()
    for tri in visible_triangles:
        edges_set.add(tuple(sorted([tri[0], tri[1]])))
        edges_set.add(tuple(sorted([tri[1], tri[2]])))
        edges_set.add(tuple(sorted([tri[2], tri[0]])))
    
    # Konwertuj na linie 3D
    lines_3d = [(vertices[e[0]], vertices[e[1]]) for e in edges_set]
    
    print(f"Wygenerowano {len(lines_3d)} linii 3D z {len(visible_triangles)} widocznych trójkątów")
    return lines_3d

# ---------------------
# Optymalizacja ścieżki robota
# ---------------------
@measure_time
def optimize_drawing_order(lines_3d, max_lines=2000):
    """Optymalizuje kolejność rysowania używając algorytmu najbliższego sąsiada"""
    if not lines_3d:
        return []
    
    # Ogranicz liczbę linii dla wydajności
    if len(lines_3d) > max_lines:
        print(f"Redukcja liczby linii z {len(lines_3d)} do {max_lines}")
        # Wybierz równomiernie rozłożone linie
        indices = np.linspace(0, len(lines_3d)-1, max_lines, dtype=int)
        lines_3d = [lines_3d[i] for i in indices]
    
    # Konwertuj do pythonowych tuple dla stabilności porównań
    lines_list = [(tuple(start), tuple(end)) for start, end in lines_3d]
    
    optimized = []
    
    # Rozpocznij od linii najbliżej początku układu
    start_point = np.array([0.0, 0.0, 0.0])
    
    min_dist = float('inf')
    best_idx = 0
    for i, (s, e) in enumerate(lines_list):
        dist = min(np.linalg.norm(np.array(s) - start_point), 
                   np.linalg.norm(np.array(e) - start_point))
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    
    current_line = lines_list.pop(best_idx)
    optimized.append((np.array(current_line[0]), np.array(current_line[1])))
    current_end = np.array(current_line[1])
    
    # Optymalizuj kolejność
    while lines_list and len(optimized) < max_lines:
        min_dist = float('inf')
        best_idx = 0
        flip = False
        
        for i, (start_tuple, end_tuple) in enumerate(lines_list):
            start_arr = np.array(start_tuple)
            end_arr = np.array(end_tuple)
            
            # Odległość do początku linii
            dist_start = np.linalg.norm(start_arr - current_end)
            if dist_start < min_dist:
                min_dist = dist_start
                best_idx = i
                flip = False
            
            # Odległość do końca linii (z odwróceniem)
            dist_end = np.linalg.norm(end_arr - current_end)
            if dist_end < min_dist:
                min_dist = dist_end
                best_idx = i
                flip = True
        
        next_line_tuple = lines_list.pop(best_idx)
        next_line_start = np.array(next_line_tuple[0])
        next_line_end = np.array(next_line_tuple[1])
        
        if flip:
            optimized.append((next_line_end, next_line_start))
            current_end = next_line_start
        else:
            optimized.append((next_line_start, next_line_end))
            current_end = next_line_end
    
    print(f"Zoptymalizowano kolejność {len(optimized)} linii")
    return optimized

# ---------------------
# Renderowanie widoku 3D
# ---------------------
@measure_time
def render_mesh_view(mesh, camera_position, look_at_point, output_path, 
                    width=1920, height=1080, lines_3d=None):
    """Renderuje widok meshu z danej perspektywy kamery"""
    try:
        # Sprawdź czy Open3D może renderować w środowisku
        os.environ['OPEN3D_CPU_RENDERING'] = 'true'
        
        # Utwórz wizualizator
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        
        # Dodaj mesh
        mesh_copy = o3d.geometry.TriangleMesh(mesh)
        mesh_copy.compute_vertex_normals()
        vis.add_geometry(mesh_copy)
        
        # Dodaj linie jeśli dostępne
        if lines_3d:
            line_set = create_line_set(lines_3d[:min(len(lines_3d), 5000)])  # Limit dla wydajności
            vis.add_geometry(line_set)
        
        # Konfiguracja kamery
        ctr = vis.get_view_control()
        
        if ctr is None:
            raise RuntimeError("View control nie został zainicjalizowany")
        
        # Renderuj aby zainicjalizować view control
        vis.poll_events()
        vis.update_renderer()
        
        # Oblicz kierunki kamery
        forward = look_at_point - camera_position
        forward = forward / np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        
        if abs(np.dot(forward, up)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        
        # Ustaw kamerę
        ctr.set_lookat(look_at_point)
        ctr.set_front(forward)
        ctr.set_up(up)
        
        # Renderuj i zapisz obraz
        vis.poll_events()
        vis.update_renderer()
        
        image = vis.capture_screen_float_buffer(do_render=True)
        image_array = (np.asarray(image) * 255).astype(np.uint8)
        
        if image_array.size == 0:
            raise RuntimeError("Przechwycenie bufora ekranu nie powiodło się")
        
        # Konwertuj BGR dla OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        
        vis.destroy_window()
        print(f"Render 3D zapisany: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Błąd renderowania 3D: {e}")
        print("Przełączanie na rendering wireframe 2D...")
        return create_wireframe_render(mesh, camera_position, look_at_point, output_path, 
                                      width, height, lines_3d)

def calculate_camera_matrix(camera_pos, look_at):
    """Oblicza macierz kamery dla Open3D"""
    forward = look_at - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # Oblicz prawy i górny wektor
    if abs(forward[2]) > 0.9:
        right = np.array([1, 0, 0])
    else:
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    
    # Macierz ekstrinsyczna
    rotation = np.column_stack([right, up, -forward])
    translation = -rotation.T @ camera_pos
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation.T
    extrinsic[:3, 3] = translation
    
    return extrinsic

def create_line_set(lines_3d):
    """Tworzy obiekt LineSet z linii 3D"""
    points = []
    lines = []
    
    for i, (start, end) in enumerate(lines_3d):
        points.extend([start, end])
        lines.append([i*2, i*2+1])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # Czerwony
    
    return line_set

def create_wireframe_render(mesh, camera_pos, look_at, output_path, width, height, lines_3d=None):
    """Tworzy wireframe render 2D w stylu Blendera"""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(vertices) == 0:
        raise ValueError("Brak wierzchołków w meshu")
    
    # Utwórz obraz z gradientem tła
    img = create_gradient_background(width, height)
    
    # Oblicz projekcję
    view_dir = look_at - camera_pos
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    proj_vertices, axis_labels = project_vertices(vertices, view_dir)
    
    # Skalowanie i przesunięcie
    scaled_vertices = scale_and_center_vertices(proj_vertices, width, height)
    
    # Rysuj trójkąty wireframe
    draw_wireframe_triangles(img, scaled_vertices, triangles)
    
    # Rysuj linie ścieżki robota
    if lines_3d:
        draw_robot_path_lines(img, lines_3d, view_dir, width, height)
    
    # Dodaj informacje i osie
    add_info_panel(img, mesh, lines_3d, axis_labels)
    add_coordinate_axes(img, view_dir)
    
    cv2.imwrite(output_path, img)
    print(f"Wireframe 2D zapisany: {output_path}")
    return output_path

def create_gradient_background(width, height):
    """Tworzy gradientowe tło"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        color_val = int(30 + (50 - 30) * (y / height))
        img[y, :] = [color_val, color_val, color_val]
    return img

def project_vertices(vertices, view_dir):
    """Projektuje wierzchołki 3D na 2D w zależności od kierunku widoku"""
    if abs(view_dir[2]) > 0.7:  # Widok z góry/dołu
        proj = vertices[:, :2]
        labels = ('X', 'Y')
    elif abs(view_dir[1]) > 0.7:  # Widok z przodu/tyłu
        proj = vertices[:, [0, 2]]
        labels = ('X', 'Z')
    else:  # Widok z boku
        proj = vertices[:, [1, 2]]
        labels = ('Y', 'Z')
    
    return proj, labels

def scale_and_center_vertices(vertices_2d, width, height, margin=0.1):
    """Skaluje i centruje wierzchołki 2D w obrazie"""
    min_vals = np.min(vertices_2d, axis=0)
    max_vals = np.max(vertices_2d, axis=0)
    ranges = max_vals - min_vals
    
    if np.any(ranges < 1e-6):
        ranges = np.where(ranges < 1e-6, 1.0, ranges)
    
    scale = min(width * (1 - 2*margin) / ranges[0], 
                height * (1 - 2*margin) / ranges[1])
    
    center = (min_vals + max_vals) / 2
    scaled = (vertices_2d - center) * scale
    
    scaled[:, 0] += width // 2
    scaled[:, 1] += height // 2
    
    return scaled.astype(int)

def draw_wireframe_triangles(img, vertices_2d, triangles):
    """Rysuje trójkąty jako wireframe"""
    triangle_count = 0
    for tri in triangles:
        if all(0 <= vertices_2d[idx][0] < img.shape[1] and 
               0 <= vertices_2d[idx][1] < img.shape[0] for idx in tri):
            pts = vertices_2d[tri]
            cv2.polylines(img, [pts], True, (180, 180, 180), 1, cv2.LINE_AA)
            triangle_count += 1
    print(f"Narysowano {triangle_count} trójkątów wireframe")

def draw_robot_path_lines(img, lines_3d, view_dir, width, height):
    """Rysuje linie ścieżki robota"""
    if not lines_3d:
        return
    
    # Ogranicz liczbę rysowanych linii dla wydajności
    max_lines_to_draw = 10000
    if len(lines_3d) > max_lines_to_draw:
        print(f"Rysowanie {max_lines_to_draw}/{len(lines_3d)} linii ścieżki robota")
        step = len(lines_3d) // max_lines_to_draw
        lines_to_draw = lines_3d[::step]
    else:
        lines_to_draw = lines_3d
    
    # Najpierw projekcja wszystkich punktów
    vertices = []
    for start_3d, end_3d in lines_to_draw:
        if abs(view_dir[2]) > 0.7:
            vertices.append(start_3d[:2])
            vertices.append(end_3d[:2])
        elif abs(view_dir[1]) > 0.7:
            vertices.append(start_3d[[0, 2]])
            vertices.append(end_3d[[0, 2]])
        else:
            vertices.append(start_3d[[1, 2]])
            vertices.append(end_3d[[1, 2]])
    
    vertices = np.array(vertices)
    
    # Skalowanie i centrowanie
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    ranges = max_vals - min_vals
    ranges = np.where(ranges < 1e-6, 1.0, ranges)
    
    margin = 0.15
    scale = min(width * (1 - 2*margin) / ranges[0], 
                height * (1 - 2*margin) / ranges[1])
    
    center = (min_vals + max_vals) / 2
    scaled = (vertices - center) * scale
    scaled[:, 0] += width // 2
    scaled[:, 1] += height // 2
    scaled = scaled.astype(int)
    
    # Rysuj linie
    line_count = 0
    for i in range(0, len(scaled), 2):
        if i+1 < len(scaled):
            start_2d = tuple(scaled[i])
            end_2d = tuple(scaled[i+1])
            
            if (0 <= start_2d[0] < width and 0 <= start_2d[1] < height and
                0 <= end_2d[0] < width and 0 <= end_2d[1] < height):
                cv2.line(img, start_2d, end_2d, (0, 0, 255), 2, cv2.LINE_AA)
                line_count += 1
    
    print(f"Narysowano {line_count} linii ścieżki robota")

# ---------------------
# Eksport ścieżek robota
# ---------------------
@measure_time
def export_robot_data(lines_3d, output_dir, name="robot_path"):
    """Eksportuje dane ścieżki robota w różnych formatach"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not lines_3d:
        print("Brak linii do eksportu")
        return None
    
    print(f"Eksport {len(lines_3d)} linii...")
    
    # Zoptymalizuj kolejność (z limitem)
    optimized_lines = optimize_drawing_order(lines_3d, max_lines=2000)
    
    # Eksport punktów
    points = np.vstack([[s, e] for s, e in optimized_lines])
    points_file = os.path.join(output_dir, f"{name}_points.txt")
    np.savetxt(points_file, points, fmt="%.6f")
    
    # Eksport linii
    lines_file = os.path.join(output_dir, f"{name}_lines.txt")
    with open(lines_file, 'w') as f:
        for s, e in optimized_lines:
            f.write(f"{s[0]:.6f} {s[1]:.6f} {s[2]:.6f} {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}\n")
    
    # Eksport komend Hanwha HCR-3A
    commands_file = export_hanwha_commands(optimized_lines, output_dir, name)
    
    print(f"Dane robota zapisane:")
    print(f"  - Punkty: {points_file}")
    print(f"  - Linie: {lines_file}")
    print(f"  - Komendy: {commands_file}")
    
    return points_file

@measure_time
def export_hanwha_commands(lines_3d, output_dir, name):
    """Generuje komendy dla robota Hanwha HCR-3A"""
    commands_file = os.path.join(output_dir, f"{name}_hanwha_commands.txt")
    stats_file = os.path.join(output_dir, f"{name}_stats.json")
    
    with open(commands_file, 'w') as f:
        f.write("# Komendy robota Hanwha HCR-3A\n")
        f.write(f"# Wygenerowano: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Liczba linii: {len(lines_3d)}\n\n")
        
        # Inicjalizacja
        f.write("INIT\n")
        f.write(f"SET_SPEED DRAW={Config.DRAWING_SPEED} TRAVEL={Config.TRAVEL_SPEED}\n")
        f.write(f"SET_Z_LIFT {Config.Z_LIFT}\n\n")
        
        # Przejdź do pozycji startowej
        first_start = lines_3d[0][0]
        f.write("# Przejdź do pozycji startowej\n")
        f.write(f"MOVE_LIFT {first_start[0]*1000:.3f} {first_start[1]*1000:.3f} {first_start[2]*1000:.3f}\n\n")
        
        # Rysuj linie
        f.write("# Rozpocznij rysowanie\n")
        f.write("START_DRAWING\n")
        
        for i, (start, end) in enumerate(lines_3d, 1):
            f.write(f"# Linia {i}\n")
            f.write(f"LINE {end[0]*1000:.3f} {end[1]*1000:.3f} {end[2]*1000:.3f}\n")
        
        # Zakończ
        f.write("\n# Zakończ rysowanie\n")
        f.write("STOP_DRAWING\n")
        f.write("RETURN_HOME\n")
    
    # Oblicz statystyki
    stats = calculate_path_statistics(lines_3d)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statystyki: {stats_file}")
    print(f"  - Całkowita długość rysowania: {stats['total_drawing_length']:.2f} mm")
    print(f"  - Całkowita długość przemieszczania: {stats['total_travel_length']:.2f} mm")
    print(f"  - Szacowany czas: {stats['estimated_total_time']:.2f} s")
    
    return commands_file

def calculate_path_statistics(lines_3d):
    """Oblicza statystyki ścieżki"""
    drawing_length = sum(np.linalg.norm(end - start) * 1000 for start, end in lines_3d)
    
    travel_length = 0
    for i in range(1, len(lines_3d)):
        prev_end = lines_3d[i-1][1]
        curr_start = lines_3d[i][0]
        travel_length += np.linalg.norm(curr_start - prev_end) * 1000
    
    drawing_time = drawing_length / Config.DRAWING_SPEED
    travel_time = travel_length / Config.TRAVEL_SPEED
    total_time = drawing_time + travel_time
    
    return {
        "total_lines": len(lines_3d),
        "total_drawing_length": drawing_length,
        "total_travel_length": travel_length,
        "estimated_drawing_time": drawing_time,
        "estimated_travel_time": travel_time,
        "estimated_total_time": total_time,
        "drawing_speed": Config.DRAWING_SPEED,
        "travel_speed": Config.TRAVEL_SPEED
    }

# ---------------------
# Wizualizacja interaktywna
# ---------------------
def interactive_visualization(mesh, lines_3d=None):
    """Interaktywna wizualizacja meshu i ścieżki"""
    geometries = [mesh]
    
    if lines_3d:
        line_set = create_line_set(lines_3d)
        geometries.append(line_set)
    
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    
    print("Otwieranie wizualizacji...")
    print("Sterowanie:")
    print("  - Lewy przycisk + przeciąganie: obracanie")
    print("  - Prawy przycisk + przeciąganie: przesuwanie")
    print("  - Kółko myszy: przybliżanie/oddalanie")
    print("  - [H] - pomoc, [Q] - wyjście")
    
    o3d.visualization.draw_geometries(geometries, window_name="Mesh + Robot Path")

# ---------------------
# Główna funkcja
# ---------------------
@measure_time
def main(point_cloud_path, camera_position=None, look_at_point=None):
    """Główna funkcja przetwarzająca chmurę punktów"""
    print("=" * 60)
    print("CHMURA PUNKTÓW -> MESH -> ŚCIEŻKA ROBOTA")
    print("=" * 60)
    
    # Wczytaj chmurę punktów
    points_data = load_point_cloud(point_cloud_path)
    print(f"Wczytano {len(points_data)} punktów")
    
    # Przetwarzanie wstępne
    clean_points = preprocess_point_cloud(points_data, voxel_size=Config.VOXEL_SIZE)
    
    # Budowa meshu
    mesh = build_mesh(clean_points, method="poisson", depth=Config.POISSON_DEPTH)
    
    # Automatyczna konfiguracja kamery jeśli nie podana
    if camera_position is None or look_at_point is None:
        bounds = mesh.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        
        if camera_position is None:
            camera_position = center + np.array([0, -extent[1]*2, extent[2]*1.5])
        if look_at_point is None:
            look_at_point = center
    
    print(f"Kamera: {camera_position}")
    print(f"Cel: {look_at_point}")
    
    # Generuj linie 3D
    lines_3d = generate_simplified_lines_from_mesh(mesh, camera_position, 
                                                  threshold=Config.VISIBILITY_THRESHOLD)
    
    # Renderuj widok
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    render_output = os.path.join(Config.OUTPUT_DIR, "mesh_render.png")
    render_mesh_view(mesh, camera_position, look_at_point, render_output, 
                    Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, lines_3d)
    
    # Eksport danych robota
    export_robot_data(lines_3d, Config.OUTPUT_DIR)
    
    # Zapisz mesh
    mesh_output = os.path.join(Config.OUTPUT_DIR, "reconstructed_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_output, mesh)
    print(f"Mesh zapisany: {mesh_output}")
    
    # Wizualizacja interaktywna
    try:
        interactive_visualization(mesh, lines_3d)
    except Exception as e:
        print(f"Błąd wizualizacji: {e}")
    
    print("Przetwarzanie zakończone!")

# ---------------------
# Entry point
# ---------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python main.py <ścieżka_do_chmury_punktów>")
        print("Opcjonalne parametry: [camera_x camera_y camera_z lookat_x lookat_y lookat_z]")
        print("\nPrzykład:")
        print("  python main.py cloud.las")
        print("  python main.py cloud.ply 1.0 2.0 3.0 0.0 0.0 0.0")
        sys.exit(1)
    
    point_cloud_path = sys.argv[1]
    
    # Parsuj opcjonalne parametry kamery
    camera_pos = None
    look_at = None
    
    if len(sys.argv) >= 8:
        try:
            camera_pos = np.array([float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])])
            look_at = np.array([float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])])
            print(f"Użyto podanej pozycji kamery: {camera_pos}")
        except ValueError:
            print("Błąd parsowania parametrów kamery - użycie wartości domyślnych")
    
    main(point_cloud_path, camera_pos, look_at)