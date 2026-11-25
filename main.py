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
    SMOOTH_ITERATIONS = 1
    # Krawędź zostaje, gdy sąsiednie ściany łamią się o co najmniej tyle stopni.
    CREASE_ANGLE_DEG = 35.0
    # Uproszczenie brzegu: punkt odpada, gdy leży bliżej cięciwy niż tyle wokseli.
    EDGE_SIMPLIFY_VOXELS = 0.5
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
    
    # Jedno przejście Taubina zamiast kilku Laplace'a: mniej kurczy powierzchnię.
    print("Wygładzanie meshu...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=Config.SMOOTH_ITERATIONS)

    # Zwijaj tylko trójkąty, których błąd jest mniejszy niż woksel.
    # Brzeg ma dużą wagę, żeby kontur nie uciekł od chmury punktów.
    if simplify and len(mesh.triangles) > 1:
        before = len(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=1,
            maximum_error=float(Config.VOXEL_SIZE ** 2),
            boundary_weight=10.0,
        )
        print(f"Uproszczenie siatki w budżecie błędu: {before} -> {len(mesh.triangles)} trójkątów")
    
    # Oblicz normalne
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    print(f"Utworzono mesh: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów")
    return mesh

# ---------------------
# Generowanie linii 3D z meshu
# ---------------------
@measure_time
def extract_feature_edges(mesh, camera_position, threshold=0.1, crease_deg=35.0):
    """Brzeg widocznej powierzchni i ostre załamania. Płaskie krawędzie wewnętrzne odpadają."""
    if len(mesh.triangles) == 0:
        print("Mesh nie zawiera trójkątów")
        return []

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(mesh.triangle_normals) == 0:
        mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)

    centers = np.mean(vertices[triangles], axis=1)
    view_dirs = camera_position - centers
    view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
    visible = np.einsum("ij,ij->i", normals, view_dirs) > threshold
    visible_triangles = triangles[visible]
    visible_normals = normals[visible]

    if len(visible_triangles) == 0:
        print("Brak widocznych trójkątów")
        return []

    edge_faces = {}
    for face_index, tri in enumerate(visible_triangles):
        corners = (int(tri[0]), int(tri[1]), int(tri[2]))
        for a, b in ((corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])):
            edge_faces.setdefault((a, b) if a < b else (b, a), []).append(face_index)

    crease = np.deg2rad(crease_deg)
    kept = []
    for (i, j), faces in edge_faces.items():
        if len(faces) == 1:
            kept.append((i, j))
            continue
        n0 = visible_normals[faces[0]]
        n1 = visible_normals[faces[1]]
        angle = np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))
        if angle >= crease:
            kept.append((i, j))

    lines_3d = [(vertices[i], vertices[j]) for i, j in kept]
    print(
        f"Krawędzie rysunku: {len(lines_3d)} z {len(edge_faces)} "
        f"({len(visible_triangles)} widocznych trójkątów, próg załamania {crease_deg:.0f}°)"
    )
    return lines_3d


def chain_segments(segments, atol=1e-5):
    """Skleja odcinki stykające się wierzchołkiem w łamane."""
    if not segments:
        return []

    def key(point):
        return tuple(np.round(np.asarray(point, dtype=float) / atol).astype(np.int64))

    unused = set(range(len(segments)))
    buckets = {}
    for index, (start, end) in enumerate(segments):
        buckets.setdefault(key(start), []).append(index)
        buckets.setdefault(key(end), []).append(index)

    polylines = []
    while unused:
        index = unused.pop()
        start, end = segments[index]
        poly = [np.asarray(start, dtype=float), np.asarray(end, dtype=float)]
        for forward in (True, False):
            while True:
                tip = poly[-1] if forward else poly[0]
                previous = poly[-2] if forward and len(poly) > 1 else (poly[1] if not forward and len(poly) > 1 else tip)
                direction = tip - previous
                direction_norm = np.linalg.norm(direction)
                best = None
                best_alignment = -2.0
                for candidate in buckets.get(key(tip), ()):
                    if candidate not in unused:
                        continue
                    a, b = segments[candidate]
                    a = np.asarray(a, dtype=float)
                    b = np.asarray(b, dtype=float)
                    if np.linalg.norm(a - tip) <= atol * 2:
                        nxt = b
                    elif np.linalg.norm(b - tip) <= atol * 2:
                        nxt = a
                    else:
                        continue
                    step = nxt - tip
                    step_norm = np.linalg.norm(step)
                    alignment = 0.0
                    if direction_norm > 1e-12 and step_norm > 1e-12:
                        alignment = float(np.dot(direction, step) / (direction_norm * step_norm))
                    if alignment > best_alignment:
                        best_alignment = alignment
                        best = (candidate, nxt)
                if best is None:
                    break
                unused.remove(best[0])
                if forward:
                    poly.append(best[1])
                else:
                    poly.insert(0, best[1])
        polylines.append(np.vstack(poly))
    print(f"Połączono odcinki w {len(polylines)} łamanych")
    return polylines


def simplify_polyline(points, epsilon):
    """Ramer–Douglas–Peucker w 3D. Zostawia punkty dalej niż epsilon od cięciwy."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points

    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start_i, end_i = stack.pop()
        if end_i <= start_i + 1:
            continue
        segment = points[end_i] - points[start_i]
        length = np.linalg.norm(segment)
        chunk = points[start_i : end_i + 1]
        if length < 1e-12:
            distances = np.linalg.norm(chunk - points[start_i], axis=1)
        else:
            distances = np.linalg.norm(np.cross(chunk - points[start_i], segment), axis=1) / length
        offset = int(np.argmax(distances))
        if distances[offset] > epsilon:
            keep[start_i + offset] = True
            stack.append((start_i, start_i + offset))
            stack.append((start_i + offset, end_i))
    return points[keep]


def order_polylines(polylines):
    """Najbliższy sąsiad po końcach łamanych, z odwróceniem kierunku."""
    if len(polylines) < 2:
        return polylines

    remaining = [np.asarray(poly, dtype=float) for poly in polylines]
    origin = np.zeros(3)
    first = min(range(len(remaining)), key=lambda i: min(np.linalg.norm(remaining[i][0] - origin), np.linalg.norm(remaining[i][-1] - origin)))
    current = remaining.pop(first)
    if np.linalg.norm(current[0]) > np.linalg.norm(current[-1]):
        current = current[::-1]
    ordered = [current]

    while remaining:
        cursor = ordered[-1][-1]
        best_index = 0
        best_distance = float("inf")
        reverse = False
        for index, poly in enumerate(remaining):
            dist_start = np.linalg.norm(poly[0] - cursor)
            dist_end = np.linalg.norm(poly[-1] - cursor)
            if dist_start < best_distance:
                best_distance = dist_start
                best_index = index
                reverse = False
            if dist_end < best_distance:
                best_distance = dist_end
                best_index = index
                reverse = True
        nxt = remaining.pop(best_index)
        if reverse:
            nxt = nxt[::-1]
        ordered.append(nxt)
    return ordered


def polylines_to_segments(polylines):
    segments = []
    for poly in polylines:
        for start, end in zip(poly, poly[1:]):
            segments.append((start, end))
    return segments

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
    
    add_info_panel(img, mesh, lines_3d, axis_labels)
    add_coordinate_axes(img, view_dir)

    cv2.imwrite(output_path, img)
    print(f"Wireframe 2D zapisany: {output_path}")
    return output_path

def add_info_panel(img, mesh, lines_3d, axis_labels):
    """Krótki opis siatki w rogu zapasowego rysunku 2D."""
    line_count = 0 if not lines_3d else len(lines_3d)
    lines = [
        f"Vertices: {len(mesh.vertices)}  Triangles: {len(mesh.triangles)}",
        f"Drawing segments: {line_count}",
        f"Axes: {axis_labels[0]} / {axis_labels[1]}",
    ]
    y = 28
    for text in lines:
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        y += 24


def add_coordinate_axes(img, view_dir):
    """Mała legenda osi w lewym dolnym rogu."""
    if abs(view_dir[2]) > 0.7:
        labels = ("X", "Y")
    elif abs(view_dir[1]) > 0.7:
        labels = ("X", "Z")
    else:
        labels = ("Y", "Z")

    origin = (48, img.shape[0] - 48)
    cv2.arrowedLine(img, origin, (origin[0] + 70, origin[1]), (80, 80, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, origin, (origin[0], origin[1] - 70), (80, 255, 80), 2, cv2.LINE_AA)
    cv2.putText(img, labels[0], (origin[0] + 76, origin[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(img, labels[1], (origin[0] - 4, origin[1] - 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)


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
def export_robot_data(polylines, output_dir, name="robot_path"):
    """Eksportuje łamane rysunku. Przerwa między niepołączonymi kreskami podnosi pisak."""
    os.makedirs(output_dir, exist_ok=True)

    if not polylines:
        print("Brak linii do eksportu")
        return None

    segments = polylines_to_segments(polylines)
    print(f"Eksport {len(polylines)} łamanych ({len(segments)} odcinków)...")

    points = np.vstack(polylines)
    points_file = os.path.join(output_dir, f"{name}_points.txt")
    np.savetxt(points_file, points, fmt="%.6f")

    lines_file = os.path.join(output_dir, f"{name}_lines.txt")
    with open(lines_file, "w") as handle:
        for start, end in segments:
            handle.write(
                f"{start[0]:.6f} {start[1]:.6f} {start[2]:.6f} "
                f"{end[0]:.6f} {end[1]:.6f} {end[2]:.6f}\n"
            )

    commands_file = export_hanwha_commands(polylines, output_dir, name)
    
    print(f"Dane robota zapisane:")
    print(f"  - Punkty: {points_file}")
    print(f"  - Linie: {lines_file}")
    print(f"  - Komendy: {commands_file}")
    
    return points_file

@measure_time
def export_hanwha_commands(polylines, output_dir, name):
    """Generuje komendy dla robota Hanwha HCR-3A. MOVE_LIFT tylko przy prawdziwej przerwie."""
    commands_file = os.path.join(output_dir, f"{name}_hanwha_commands.txt")
    stats_file = os.path.join(output_dir, f"{name}_stats.json")
    lift_gap = Config.VOXEL_SIZE * 0.25
    commands = drawing_commands(polylines, lift_gap)

    with open(commands_file, "w") as handle:
        handle.write("# Komendy robota Hanwha HCR-3A\n")
        handle.write(f"# Wygenerowano: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"# Liczba łamanych: {len(polylines)}\n\n")
        handle.write("INIT\n")
        handle.write(f"SET_SPEED DRAW={Config.DRAWING_SPEED} TRAVEL={Config.TRAVEL_SPEED}\n")
        handle.write(f"SET_Z_LIFT {Config.Z_LIFT}\n\n")
        handle.write("START_DRAWING\n")
        for kind, point in commands:
            coords = f"{point[0]*1000:.3f} {point[1]*1000:.3f} {point[2]*1000:.3f}"
            if kind == "lift":
                handle.write(f"MOVE_LIFT {coords}\n")
            else:
                handle.write(f"LINE {coords}\n")
        handle.write("\nSTOP_DRAWING\n")
        handle.write("RETURN_HOME\n")

    stats = calculate_path_statistics(polylines, lift_gap)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statystyki: {stats_file}")
    print(f"  - Całkowita długość rysowania: {stats['total_drawing_length']:.2f} mm")
    print(f"  - Całkowita długość przemieszczania: {stats['total_travel_length']:.2f} mm")
    print(f"  - Szacowany czas: {stats['estimated_total_time']:.2f} s")
    
    return commands_file

def drawing_commands(polylines, lift_gap):
    """Punkty łamanej to LINE. MOVE_LIFT tylko przy starcie nowej, niepołączonej kreski."""
    commands = []
    pen = None
    for poly in polylines:
        points = np.asarray(poly, dtype=float)
        if len(points) == 0:
            continue
        if pen is None or np.linalg.norm(points[0] - pen) > lift_gap:
            commands.append(("lift", points[0]))
        pen = points[0]
        for point in points[1:]:
            if np.linalg.norm(point - pen) > 1e-6:
                commands.append(("line", point))
            pen = point
    return commands


def calculate_path_statistics(polylines, lift_gap):
    """Długość rysowania i przejazdów w milimetrach."""
    drawing_length = 0.0
    travel_length = 0.0
    pen = None
    segments = 0
    for poly in polylines:
        points = np.asarray(poly, dtype=float)
        if len(points) == 0:
            continue
        if pen is not None:
            gap = np.linalg.norm(points[0] - pen) * 1000
            if gap > lift_gap * 1000:
                travel_length += gap
        for start, end in zip(points, points[1:]):
            drawing_length += np.linalg.norm(end - start) * 1000
            segments += 1
        pen = points[-1]

    drawing_time = drawing_length / Config.DRAWING_SPEED
    travel_time = travel_length / Config.TRAVEL_SPEED
    return {
        "total_polylines": len(polylines),
        "total_lines": segments,
        "total_drawing_length": drawing_length,
        "total_travel_length": travel_length,
        "estimated_drawing_time": drawing_time,
        "estimated_travel_time": travel_time,
        "estimated_total_time": drawing_time + travel_time,
        "drawing_speed": Config.DRAWING_SPEED,
        "travel_speed": Config.TRAVEL_SPEED,
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
    
    segments = extract_feature_edges(
        mesh,
        camera_position,
        threshold=Config.VISIBILITY_THRESHOLD,
        crease_deg=Config.CREASE_ANGLE_DEG,
    )
    polylines = chain_segments(segments)
    epsilon = Config.VOXEL_SIZE * Config.EDGE_SIMPLIFY_VOXELS
    polylines = [simplify_polyline(poly, epsilon) for poly in polylines]
    polylines = [poly for poly in polylines if len(poly) >= 2]
    polylines = order_polylines(polylines)
    lines_3d = polylines_to_segments(polylines)
    print(f"Ścieżka rysunku: {len(polylines)} łamanych, {len(lines_3d)} odcinków")

    # Renderuj widok
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    render_output = os.path.join(Config.OUTPUT_DIR, "mesh_render.png")
    render_mesh_view(mesh, camera_position, look_at_point, render_output, 
                    Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, lines_3d)
    
    # Eksport danych robota
    export_robot_data(polylines, Config.OUTPUT_DIR)
    
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