#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import open3d as o3d
import time
from functools import wraps
import os
import json
import argparse
from collections import Counter, defaultdict
import svgwrite
import xml.etree.ElementTree as ET

# Opcjonalne: Numba dla przyspieszenia
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        return lambda func: func
    prange = range

# ---------------------
# Konfiguracja
# ---------------------
class Config:
    OUTPUT_DIR = "./output"
    VISIBILITY_THRESHOLD = 0.1
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080
    DRAWING_SPEED = 100
    TRAVEL_SPEED = 150
    MAX_TRIANGLES = 200
    PAPER_FORMAT = "A4"
    ROBOT_WORKSPACE_Z = 100.0
    ROBOT_SAFE_Z = 50.0
    PAPER_SIZES = {"A4": (210.0, 297.0), "A5": (148.0, 210.0), "A3": (297.0, 420.0)}
    PAPER_MARGIN = 10.0
    PAPER_LANDSCAPE = False

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[BENCH] {func.__name__}: {time.time() - start:.4f}s")
        return result
    return wrapper

# ---------------------
# Funkcje Numba
# ---------------------
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def compute_backface_culling_numba(centers, normals, camera_pos, threshold):
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
        best_idx, best_dist, flip = -1, np.inf, False
        for i in range(len(starts)):
            if used_mask[i]:
                continue
            d = current_end - starts[i]
            dist_start = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
            if dist_start < best_dist:
                best_dist, best_idx, flip = dist_start, i, False
            d = current_end - ends[i]
            dist_end = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
            if dist_end < best_dist:
                best_dist, best_idx, flip = dist_end, i, True
        return best_idx, flip

# ---------------------
# Pomocnicze funkcje
# ---------------------
def backface_culling(centers, normals, camera_pos, threshold):
    """Backface culling - Numba lub NumPy"""
    if NUMBA_AVAILABLE:
        return compute_backface_culling_numba(centers, normals, camera_pos.astype(np.float32), threshold)
    view_dirs = camera_pos - centers
    norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return np.einsum('ij,ij->i', normals, view_dirs / norms) > threshold

def get_projection_axes(view_dir):
    """Zwraca indeksy osi dla projekcji 2D"""
    if abs(view_dir[2]) > 0.7:
        return [0, 1]
    elif abs(view_dir[1]) > 0.7:
        return [0, 2]
    return [1, 2]

def chain_segments(edges, vertices):
    """Łączy krawędzie w ciągłe łańcuchy"""
    adj = defaultdict(list)
    for v1, v2 in edges:
        adj[v1].append(v2)
        adj[v2].append(v1)
    
    chains = []
    while adj:
        start = next((n for n in adj if len(adj[n]) == 1), next(iter(adj)))
        chain = [vertices[start].copy()]
        curr = start
        
        while curr in adj and adj[curr]:
            next_node = adj[curr].pop(0)
            if not adj[curr]:
                del adj[curr]
            if next_node in adj and curr in adj[next_node]:
                adj[next_node].remove(curr)
                if not adj[next_node]:
                    del adj[next_node]
            chain.append(vertices[next_node].copy())
            if next_node == start:
                break
            curr = next_node
        
        if len(chain) > 1:
            chains.append(chain)
    return chains

def optimize_chain_order(chains):
    """Optymalizuje kolejność łańcuchów"""
    if not chains:
        return []
    ordered, remaining = [], chains[:]
    current_pos = np.zeros(3)
    
    while remaining:
        best_idx, best_dist, best_rev = 0, float('inf'), False
        for i, c in enumerate(remaining):
            d_start = np.sum((current_pos - c[0])**2)
            d_end = np.sum((current_pos - c[-1])**2)
            if d_start < best_dist:
                best_idx, best_dist, best_rev = i, d_start, False
            if d_end < best_dist:
                best_idx, best_dist, best_rev = i, d_end, True
        
        chain = remaining.pop(best_idx)
        if best_rev:
            chain = chain[::-1]
        ordered.append(chain)
        current_pos = chain[-1]
    return ordered

# ---------------------
# Generowanie linii
# ---------------------
@measure_time
def generate_lines(mesh, camera_pos, threshold=0.1, wireframe=False):
    """Generuje linie z meshu"""
    if len(mesh.triangles) == 0:
        return []

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    
    if len(mesh.triangle_normals) == 0:
        mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals, dtype=np.float32)
    centers = np.mean(vertices[triangles], axis=1).astype(np.float32)
    
    visible_mask = backface_culling(centers, normals, camera_pos, threshold)
    visible_tris = triangles[visible_mask]
    
    if len(visible_tris) == 0:
        return []
    
    # Zlicz krawędzie
    edge_counter = Counter()
    for tri in visible_tris:
        for i in range(3):
            edge_counter[tuple(sorted([tri[i], tri[(i+1)%3]]))] += 1
    
    if wireframe:
        edges = list(edge_counter.keys())
    else:
        edges = [e for e, c in edge_counter.items() if c == 1]
        chains = chain_segments(edges, vertices)
        chains = optimize_chain_order(chains)
        return [(c[i], c[i+1]) for c in chains for i in range(len(c)-1)]
    
    return [(vertices[e[0]], vertices[e[1]]) for e in edges]

@measure_time
def optimize_path(lines, max_lines=2000):
    """Optymalizuje kolejność linii"""
    if not lines or len(lines) <= 1:
        return lines
    
    if len(lines) > max_lines:
        lines = [lines[i] for i in np.linspace(0, len(lines)-1, max_lines, dtype=int)]
    
    starts = np.array([l[0] for l in lines], dtype=np.float32)
    ends = np.array([l[1] for l in lines], dtype=np.float32)
    
    optimized, used = [lines[0]], np.zeros(len(lines), dtype=bool)
    used[0] = True
    current_end = ends[0]
    
    for _ in range(len(lines) - 1):
        if NUMBA_AVAILABLE:
            best_idx, flip = find_nearest_numba(current_end, starts, ends, used)
        else:
            unused = np.where(~used)[0]
            if len(unused) == 0:
                break
            d_starts = np.linalg.norm(starts[unused] - current_end, axis=1)
            d_ends = np.linalg.norm(ends[unused] - current_end, axis=1)
            if np.min(d_starts) < np.min(d_ends):
                best_idx, flip = unused[np.argmin(d_starts)], False
            else:
                best_idx, flip = unused[np.argmin(d_ends)], True
        
        if best_idx == -1:
            break
        used[best_idx] = True
        line = lines[best_idx]
        if flip:
            line = (line[1], line[0])
        optimized.append(line)
        current_end = line[1]
    
    return optimized

# ---------------------
# Renderowanie
# ---------------------
@measure_time
def render_png(lines, output_path, view_dir):
    """Renderuje linie do PNG"""
    img = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3), dtype=np.uint8)
    img[:] = 40
    
    if not lines:
        cv2.imwrite(output_path, img)
        return
    
    axes = get_projection_axes(view_dir)
    pts_2d = np.array([[l[0][axes], l[1][axes]] for l in lines]).reshape(-1, 2)
    
    min_v, max_v = pts_2d.min(axis=0), pts_2d.max(axis=0)
    ranges = np.maximum(max_v - min_v, 1e-3)
    scale = min(Config.IMAGE_WIDTH * 0.8 / ranges[0], Config.IMAGE_HEIGHT * 0.8 / ranges[1])
    center = (min_v + max_v) / 2
    offset = np.array([Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT]) / 2
    
    for s, e in lines:
        p1 = ((s[axes] - center) * scale + offset).astype(int)
        p2 = ((e[axes] - center) * scale + offset).astype(int)
        cv2.line(img, tuple(p1), tuple(p2), (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(output_path, img)
    print(f"PNG: {output_path}")

# ---------------------
# Logo paths
# ---------------------
LOGO_PATHS = {
    "min": os.path.join(os.path.dirname(__file__), "logos", "min.svg"),
    "full": os.path.join(os.path.dirname(__file__), "logos", "full.svg"),
}

def get_svg_dimensions(svg_path):
    """Pobiera wymiary SVG"""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Próbuj viewBox
        viewbox = root.get('viewBox')
        if viewbox:
            parts = viewbox.split()
            if len(parts) == 4:
                return float(parts[2]), float(parts[3])
        
        # Próbuj width/height
        w = root.get('width', '100')
        h = root.get('height', '100')
        w = float(''.join(c for c in w if c.isdigit() or c == '.') or '100')
        h = float(''.join(c for c in h if c.isdigit() or c == '.') or '100')
        return w, h
    except:
        return 100, 100

def copy_element(elem):
    """Rekurencyjnie kopiuje element XML usuwając namespace"""
    # Usuń namespace z tagu
    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
    
    # Skopiuj atrybuty (usuń namespace z kluczy)
    attrib = {}
    for k, v in elem.attrib.items():
        key = k.split('}')[-1] if '}' in k else k
        attrib[key] = v
    
    new_elem = ET.Element(tag, attrib)
    new_elem.text = elem.text
    new_elem.tail = elem.tail
    
    # Rekurencyjnie kopiuj dzieci
    for child in elem:
        new_elem.append(copy_element(child))
    
    return new_elem

def embed_logo_in_svg(svg_path, logo_type, drawing_bounds):
    """Osadza logo w SVG - w rogu kartki"""
    logo_path = LOGO_PATHS.get(logo_type)
    if not logo_path or not os.path.exists(logo_path):
        print(f"Logo nie znalezione: {logo_path}")
        return
    
    # Zarejestruj namespace SVG
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    
    # Parsuj główny SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Pobierz wymiary papieru
    paper_w, paper_h = Config.PAPER_SIZES[Config.PAPER_FORMAT]
    if Config.PAPER_LANDSCAPE:
        paper_w, paper_h = paper_h, paper_w
    
    # Pobierz wymiary logo
    logo_w, logo_h = get_svg_dimensions(logo_path)
    
    margin = Config.PAPER_MARGIN
    min_y, max_y = drawing_bounds
    
    # Stały rozmiar logo (dopasowany do rogu)
    max_logo_w = 100  # mm
    max_logo_h = 50  # mm
    
    if logo_type == "min":
        max_logo_w *= .75 
        max_logo_h *= .75
    
    logo_scale = min(max_logo_w / logo_w, max_logo_h / logo_h)
    scaled_logo_w = logo_w * logo_scale
    scaled_logo_h = logo_h * logo_scale
    
    # Sprawdź który róg jest wolny
    # Preferuj prawy dolny róg
    corners = [
        ("bottom_right", paper_w - margin - scaled_logo_w, paper_h - margin - scaled_logo_h),
        ("bottom_left", margin, paper_h - margin - scaled_logo_h),
        ("top_right", paper_w - margin - scaled_logo_w, margin),
        ("top_left", margin, margin),
    ]
    
    # Wybierz pierwszy wolny róg (nie kolidujący z rysunkiem)
    logo_x, logo_y = corners[0][1], corners[0][2]  # domyślnie prawy dolny
    
    for corner_name, cx, cy in corners:
        # Sprawdź czy róg nie koliduje z rysunkiem
        if cy + scaled_logo_h < min_y - 3 or cy > max_y + 3:
            logo_x, logo_y = cx, cy
            break
    
    # Parsuj logo SVG
    logo_tree = ET.parse(logo_path)
    logo_root = logo_tree.getroot()
    
    # Oblicz stroke-width
    stroke_width = max(0.3, min(1.0 / logo_scale * 0.4, 1.5))
    
    # Stwórz grupę z transformacją
    g = ET.SubElement(root, 'g')
    g.set('transform', f'translate({logo_x:.2f},{logo_y:.2f}) scale({logo_scale:.6f})')
    g.set('id', 'logo')
    g.set('fill', 'none')
    g.set('stroke', 'black')
    g.set('stroke-width', f'{stroke_width:.2f}')
    
    # Kopiuj wszystkie dzieci logo do grupy
    for child in logo_root:
        copied = copy_element_outline(child)
        g.append(copied)
    
    # Zapisz
    with open(svg_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    print(f"Logo '{logo_type}' w rogu ({scaled_logo_w:.1f}x{scaled_logo_h:.1f}mm)")

def copy_element_outline(elem):
    """Rekurencyjnie kopiuje element XML usuwając namespace i wymuszając outline"""
    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
    
    # Skopiuj atrybuty (usuń namespace z kluczy)
    attrib = {}
    for k, v in elem.attrib.items():
        key = k.split('}')[-1] if '}' in k else k
        # Usuń atrybuty fill (wymuszamy outline)
        if key.lower() == 'fill':
            continue
        # Zachowaj stroke jeśli jest
        attrib[key] = v
    
    new_elem = ET.Element(tag, attrib)
    new_elem.text = elem.text
    new_elem.tail = elem.tail
    
    # Rekurencyjnie kopiuj dzieci
    for child in elem:
        new_elem.append(copy_element_outline(child))
    
    return new_elem

def render_svg(lines, output_path, view_dir, logo_type=None):
    """Renderuje linie do SVG używając svgwrite"""
    paper_w, paper_h = Config.PAPER_SIZES[Config.PAPER_FORMAT]
    if Config.PAPER_LANDSCAPE:
        paper_w, paper_h = paper_h, paper_w
    
    dwg = svgwrite.Drawing(output_path, size=(f"{paper_w}mm", f"{paper_h}mm"),
                           viewBox=f"0 0 {paper_w} {paper_h}")
    
    drawing_bounds = (paper_h / 2, paper_h / 2)  # domyślne
    
    if lines:
        axes = get_projection_axes(view_dir)
        pts_2d = np.array([[l[0][axes], l[1][axes]] for l in lines]).reshape(-1, 2)
        
        min_v, max_v = pts_2d.min(axis=0), pts_2d.max(axis=0)
        ranges = np.maximum(max_v - min_v, 1e-3)
        margin = Config.PAPER_MARGIN
        scale = min((paper_w - 2*margin) / ranges[0], (paper_h - 2*margin) / ranges[1])
        center = (min_v + max_v) / 2
        offset = np.array([paper_w, paper_h]) / 2
        
        g = dwg.g(stroke="black", stroke_width="0.5", fill="none")
        
        y_coords = []
        for s, e in lines:
            p1 = (s[axes] - center) * scale + offset
            p2 = (e[axes] - center) * scale + offset
            g.add(dwg.line(start=(p1[0], p1[1]), end=(p2[0], p2[1])))
            y_coords.extend([p1[1], p2[1]])
        
        dwg.add(g)
        
        if y_coords:
            drawing_bounds = (min(y_coords), max(y_coords))
    
    dwg.save()
    print(f"SVG: {output_path}")
    
    # Dodaj logo jeśli podano
    if logo_type:
        embed_logo_in_svg(output_path, logo_type, drawing_bounds)

# ---------------------
# Eksport i skalowanie
# ---------------------
@measure_time
def load_and_scale_mesh(path):
    """Wczytuje i skaluje mesh"""
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > Config.MAX_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(Config.MAX_TRIANGLES)
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    paper_w, paper_h = Config.PAPER_SIZES[Config.PAPER_FORMAT]
    if Config.PAPER_LANDSCAPE:
        paper_w, paper_h = paper_h, paper_w
    
    bbox = mesh.get_axis_aligned_bounding_box()
    extent, center = bbox.get_extent(), bbox.get_center()
    
    usable = np.array([paper_w, paper_h]) - 2 * Config.PAPER_MARGIN
    max_z = Config.ROBOT_WORKSPACE_Z - Config.ROBOT_SAFE_Z - 10
    scale = min(usable[0]/max(extent[0], 1e-6), usable[1]/max(extent[1], 1e-6), max_z/max(extent[2], 1e-6))
    
    vertices = (np.asarray(mesh.vertices) - center) * scale
    vertices[:, :2] += [paper_w/2, paper_h/2]
    vertices[:, 2] += Config.ROBOT_SAFE_Z + extent[2]*scale/2
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_triangle_normals()
    
    print(f"Mesh: {len(mesh.vertices)} wierz., {len(mesh.triangles)} trój., skala: {scale:.4f}")
    return mesh

# ---------------------
# Main
# ---------------------
def main(mesh_path, camera_pos=None, look_at=None, dev_mode=False, save_mesh=False, logo_type=None):
    print("=" * 50)
    print("MESH -> ROBOT PATH CONVERTER")
    print("=" * 50)
    
    mesh = load_and_scale_mesh(mesh_path)
    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    if dev_mode:
        png_dir = os.path.join(Config.OUTPUT_DIR, "png")
        svg_dir = os.path.join(Config.OUTPUT_DIR, "svg")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(svg_dir, exist_ok=True)
        
        safe_dist = max(extent[0], extent[1]) * 2.5
        max_ext = np.max(extent)
        positions = [
            ("front", np.array([0, -safe_dist, max_ext])),
            ("side", np.array([safe_dist, 0, max_ext])),
            ("corner", np.array([safe_dist*0.7, -safe_dist*0.7, max_ext*1.5])),
            ("top", np.array([0, 0, safe_dist])),
        ]
        
        for name, offset in positions:
            cam = center + offset
            view_dir = (center - cam) / np.linalg.norm(center - cam)
            
            for mode in ["outline", "wireframe"]:
                lines = generate_lines(mesh, cam, Config.VISIBILITY_THRESHOLD, mode == "wireframe")
                lines = optimize_path(lines)
                
                render_png(lines, os.path.join(png_dir, f"{name}_{mode}.png"), view_dir)
                render_svg(lines, os.path.join(svg_dir, f"{name}_{mode}.svg"), view_dir, logo_type)
    else:
        if camera_pos is None:
            dist = max(extent[0], extent[1]) * 2.5
            camera_pos = center + np.array([0, 0, dist])
            look_at = center
            R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi))
            mesh.rotate(R, center=center)
        
        view_dir = (look_at - camera_pos) / np.linalg.norm(look_at - camera_pos)
        
        for mode in ["outline", "wireframe"]:
            lines = generate_lines(mesh, camera_pos, Config.VISIBILITY_THRESHOLD, mode == "wireframe")
            lines = optimize_path(lines)
            
            render_png(lines, os.path.join(Config.OUTPUT_DIR, f"{mode}.png"), view_dir)
            render_svg(lines, os.path.join(Config.OUTPUT_DIR, f"{mode}.svg"), view_dir, logo_type)
    
    if save_mesh:
        o3d.io.write_triangle_mesh(os.path.join(Config.OUTPUT_DIR, "mesh.ply"), mesh)
        print(f"Mesh zapisany: {Config.OUTPUT_DIR}/mesh.ply")
    
    print(f"\n✅ Gotowe! -> {Config.OUTPUT_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mesh -> Robot Path')
    parser.add_argument('mesh_file', help='Plik meshu (.ply, .obj, .stl)')
    parser.add_argument('--camera', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--target', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--dev', action='store_true', help='Tryb deweloperski')
    parser.add_argument('--save-mesh', action='store_true', help='Zapisz przetworzony mesh do PLY')
    parser.add_argument('--logo', choices=['min', 'full'], help='Dodaj logo (min lub full)')
    args = parser.parse_args()
    
    cam = np.array(args.camera) if args.camera else None
    target = np.array(args.target) if args.target else None
    
    if (cam is None) != (target is None):
        sys.exit("Błąd: Podaj --camera i --target razem")
    
    main(args.mesh_file, cam, target, args.dev, args.save_mesh, args.logo)