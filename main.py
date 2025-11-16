#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import open3d as o3d
import time
from functools import wraps, lru_cache
import os
import json

# Opcjonalne biblioteki
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
    VOXEL_SIZE = 0.02
    POISSON_DEPTH = 9
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
# Przetwarzanie chmury punktów
# ---------------------
@measure_time
def preprocess_point_cloud(points, voxel_size=0.02):
    """Czyszczenie i downsampling"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Usuwanie outlierów
    pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Usunięto {len(points) - len(inliers)} outlierów")
    
    # Downsampling
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Po downsamplingu: {len(pcd.points)} punktów")
    
    return np.asarray(pcd.points, dtype=np.float32)

# ---------------------
# Budowa meshu (uproszczona)
# ---------------------
@measure_time
def build_mesh(points, depth=9):
    """Buduje mesh z Poisson reconstruction"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estymacja normalnych
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)
    
    # Poisson Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=True)
    
    # Filtrowanie gęstości
    if len(densities) > 0:
        densities = np.asarray(densities)
        vertices_to_keep = densities > np.quantile(densities, 0.05)
        mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    
    # Uproszczenie
    if len(mesh.triangles) > 50000:
        target = max(10000, len(mesh.triangles) // 4)
        mesh = mesh.simplify_quadric_decimation(target)
    
    # Wygładzanie
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    print(f"Mesh: {len(mesh.vertices)} wierzchołków, {len(mesh.triangles)} trójkątów")
    return mesh

# ---------------------
# Generowanie linii 3D
# ---------------------
@measure_time
def generate_lines_from_mesh(mesh, camera_position, threshold=0.1):
    """Generuje linie z widocznych krawędzi"""
    if len(mesh.triangles) == 0:
        return []

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(mesh.triangle_normals) == 0:
        mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    
    # Backface culling
    centers = np.mean(vertices[triangles], axis=1)
    view_dirs = camera_position - centers
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)
    
    dot_products = np.einsum('ij,ij->i', normals, view_dirs)
    visible_triangles = triangles[dot_products > threshold]
    
    if len(visible_triangles) == 0:
        return []
    
    # Zbierz krawędzie
    edges = set()
    for tri in visible_triangles:
        edges.add(tuple(sorted([tri[0], tri[1]])))
        edges.add(tuple(sorted([tri[1], tri[2]])))
        edges.add(tuple(sorted([tri[2], tri[0]])))
    
    lines_3d = [(vertices[e[0]], vertices[e[1]]) for e in edges]
    print(f"Wygenerowano {len(lines_3d)} linii 3D")
    return lines_3d

# ---------------------
# Optymalizacja ścieżki (prosta)
# ---------------------
@measure_time
def optimize_path(lines_3d, max_lines=2000):
    """Prosta optymalizacja nearest neighbor"""
    if not lines_3d or len(lines_3d) <= 1:
        return lines_3d
    
    # Ogranicz liczbę linii
    if len(lines_3d) > max_lines:
        indices = np.linspace(0, len(lines_3d)-1, max_lines, dtype=int)
        lines_3d = [lines_3d[i] for i in indices]
    
    optimized = []
    remaining = list(lines_3d)
    
    # Rozpocznij od pierwszej linii
    current = remaining.pop(0)
    optimized.append(current)
    current_end = current[1]
    
    # Greedy nearest neighbor
    while remaining:
        min_dist = float('inf')
        best_idx = 0
        flip = False
        
        for idx, (start, end) in enumerate(remaining):
            dist_start = np.linalg.norm(current_end - start)
            dist_end = np.linalg.norm(current_end - end)
            
            if dist_start < min_dist:
                min_dist = dist_start
                best_idx = idx
                flip = False
            
            if dist_end < min_dist:
                min_dist = dist_end
                best_idx = idx
                flip = True
        
        next_line = remaining.pop(best_idx)
        if flip:
            next_line = (next_line[1], next_line[0])
        
        optimized.append(next_line)
        current_end = next_line[1]
    
    print(f"Zoptymalizowano {len(optimized)} linii")
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
# Eksport danych robota (zaktualizowany)
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
    
    # Statystyki
    total_length = sum(np.linalg.norm(e - s) * 1000 for s, e in lines_3d)
    
    # Oblicz dystans podróży (bez rysowania)
    travel_distance = 0
    for i in range(1, len(lines_3d)):
        prev_end = lines_3d[i-1][1]
        curr_start = lines_3d[i][0]
        travel_distance += np.linalg.norm(curr_start - prev_end) * 1000
    
    drawing_time = total_length / Config.DRAWING_SPEED
    travel_time = travel_distance / Config.TRAVEL_SPEED
    
    stats = {
        "total_lines": len(lines_3d),
        "total_drawing_length_mm": round(total_length, 2),
        "total_travel_distance_mm": round(travel_distance, 2),
        "estimated_drawing_time_s": round(drawing_time, 2),
        "estimated_travel_time_s": round(travel_time, 2),
        "estimated_total_time_s": round(drawing_time + travel_time, 2),
        "estimated_total_time_min": round((drawing_time + travel_time) / 60, 2),
        "drawing_speed_mm_s": Config.DRAWING_SPEED,
        "travel_speed_mm_s": Config.TRAVEL_SPEED
    }
    
    with open(os.path.join(output_dir, "robot_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Eksportowano dane robota:")
    print(f"  - {len(lines_3d)} linii")
    print(f"  - {total_length:.2f}mm rysowania")
    print(f"  - {travel_distance:.2f}mm podróży")
    print(f"  - ~{(drawing_time + travel_time)/60:.1f} min czasu")

# ---------------------
# Główna funkcja (zaktualizowana)
# ---------------------
@measure_time
def main(point_cloud_path, camera_position=None, look_at_point=None):
    """Główna funkcja"""
    print("=" * 60)
    print("UPROSZCZONA WERSJA: CHMURA -> MESH -> ROBOT PATH")
    print("=" * 60)
    
    # Wczytaj punkty
    points = load_point_cloud(point_cloud_path)
    print(f"Wczytano {len(points)} punktów")
    
    # Przetwórz
    clean_points = preprocess_point_cloud(points, Config.VOXEL_SIZE)
    
    # Zbuduj mesh
    mesh = build_mesh(clean_points, Config.POISSON_DEPTH)
    
    # Auto kamera
    if camera_position is None:
        bounds = mesh.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        camera_position = center + np.array([0, -extent[1]*2, extent[2]*1.5])
        look_at_point = center
    
    print(f"Kamera: {camera_position}")
    
    # Generuj linie
    lines_3d = generate_lines_from_mesh(mesh, camera_position, Config.VISIBILITY_THRESHOLD)
    
    # Optymalizuj
    optimized_lines = optimize_path(lines_3d, max_lines=2000)
    
    # Renderuj (z poprawką)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    render_mesh(mesh, camera_position, look_at_point, 
                os.path.join(Config.OUTPUT_DIR, "render.png"), optimized_lines)
    
    # Eksportuj dane robota (zamiast G-code)
    export_robot_data(optimized_lines, Config.OUTPUT_DIR)
    export_hanwha_commands(optimized_lines, Config.OUTPUT_DIR)
    
    # Zapisz mesh
    o3d.io.write_triangle_mesh(os.path.join(Config.OUTPUT_DIR, "mesh.ply"), mesh)
    print(f"Mesh zapisany: {Config.OUTPUT_DIR}/mesh.ply")
    
    print("Gotowe!")

# ---------------------
# Entry point
# ---------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python main_simplified.py <plik.ply>")
        sys.exit(1)
    
    main(sys.argv[1])