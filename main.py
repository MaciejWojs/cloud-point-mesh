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
import math
import concurrent.futures

# Opcjonalne: Numba dla przyspieszenia
try:
    from numba import jit, prange, njit
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
    MAX_TRIANGLES_OUTLINE = 5000  # Więcej trójkątów dla outline dla lepszej ciągłości
    PAPER_FORMAT = "A4"
    ROBOT_WORKSPACE_Z = 100.0
    ROBOT_SAFE_Z = 50.0
    PAPER_SIZES = {"A4": (210.0, 297.0), "A5": (148.0, 210.0), "A3": (297.0, 420.0)}
    PAPER_MARGIN = 10.0
    PAPER_LANDSCAPE = False
    
    # Nowe parametry progowe
    MIN_SEGMENT_LENGTH = 0.8   # mm - minimalna długość segmentu, krótsze usuwamy
    MIN_CHAIN_LENGTH = 4.0     # mm - minimalna długość całego łańcucha (sumarycznie)
    MIN_SEGMENTS_IN_CHAIN = 2  # minimalna liczba segmentów w łańcuchu

# Threshold dla wielkich siatek (liczba trójkątów)
LARGE_MESH_TRI = 1_000_000

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
    @njit(cache=True, fastmath=True)
    def compute_backface_culling_numba(centers, normals, camera_pos, threshold):
        """Numba-accelerated backface culling (szybciej)."""
        n = centers.shape[0]
        out = np.zeros(n, dtype=np.bool_)
        cx = camera_pos[0]; cy = camera_pos[1]; cz = camera_pos[2]
        for i in range(n):
            dx = cx - centers[i, 0]
            dy = cy - centers[i, 1]
            dz = cz - centers[i, 2]
            dot = normals[i, 0] * dx + normals[i, 1] * dy + normals[i, 2] * dz
            out[i] = dot > threshold
        return out

    @jit(nopython=True, cache=True)
    def find_nearest_numba(current_end, starts, ends, used_mask):
        best_idx, best_dist, flip = -1, np.inf, False
        for i in prange(len(starts)):
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
# Pomocnicze funkcje (zastąpione szybszymi implementacjami)
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


def _lines_to_graph(lines, quant=4):
    """Konwertuje listę linii (s,e) -> macierz wierzchołków i lista krawędzi (indeksy)"""
    coord_to_idx = {}
    vertices = []
    edges = []
    for s, e in lines:
        ks = tuple(np.round(s, quant).tolist())
        ke = tuple(np.round(e, quant).tolist())
        if ks not in coord_to_idx:
            coord_to_idx[ks] = len(vertices); vertices.append(np.array(ks, dtype=np.float64))
        if ke not in coord_to_idx:
            coord_to_idx[ke] = len(vertices); vertices.append(np.array(ke, dtype=np.float64))
        edges.append((coord_to_idx[ks], coord_to_idx[ke]))
    return np.array(vertices, dtype=np.float64), edges

def _edge_length(vertices, edge):
    a, b = edge
    return float(np.linalg.norm(vertices[a] - vertices[b]))

def _prune_short_dangles(vertices, edges, min_len, max_iter=10):
    """
    Iteracyjnie usuwa krawędzie wiszące (degree==1) krótsze niż min_len.
    Zwraca listę pozostałych krawędzi jako tuple(index,index).
    """
    adj = defaultdict(list)
    for a,b in edges:
        adj[a].append(b); adj[b].append(a)
    edge_set = set(tuple(sorted(e)) for e in edges)
    for _ in range(max_iter):
        to_remove = []
        degrees = {v: len(nei) for v, nei in adj.items()}
        for e in list(edge_set):
            a,b = e
            if degrees.get(a,0) == 1 or degrees.get(b,0) == 1:
                if _edge_length(vertices, e) < min_len:
                    to_remove.append(e)
        if not to_remove:
            break
        for e in to_remove:
            a,b = e
            edge_set.discard(e)
            if b in adj[a]: adj[a].remove(b)
            if a in adj[b]: adj[b].remove(a)
    return [tuple(e) for e in edge_set]

def chain_segments(edges, vertices):
    """Łączy krawędzie w ciągłe łańcuchy - ulepszona wersja dla lepszej ciągłości"""
    if not edges:
        return []
    
    # Buduj graf sąsiedztwa
    adj = defaultdict(list)
    for v1, v2 in edges:
        adj[v1].append(v2)
        adj[v2].append(v1)
    
    chains = []
    visited_edges = set()
    
    # Znajdź wierzchołki o nieparzystym stopniu (końce łańcuchów)
    odd_vertices = [v for v, neighbors in adj.items() if len(neighbors) % 2 == 1]
    
    # Jeśli nie ma wierzchołków o nieparzystym stopniu, wybierz dowolny
    start_vertices = odd_vertices if odd_vertices else list(adj.keys())
    
    for start in start_vertices[:]:
        if start not in adj or not adj[start]:
            continue
            
        # Użyj DFS dla lepszej ciągłości
        stack = [start]
        chain = []
        
        while stack:
            curr = stack[-1]
            if adj[curr]:
                next_node = adj[curr].pop()
                if next_node in adj and curr in adj[next_node]:
                    adj[next_node].remove(curr)
                
                edge_key = tuple(sorted([curr, next_node]))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    stack.append(next_node)
            else:
                chain.append(stack.pop())
        
        if len(chain) > 1:
            # Konwertuj indeksy wierzchołków na współrzędne
            coord_chain = [vertices[i] for i in chain]
            chains.append(coord_chain)
    
    # Dodaj pozostałe cykle
    remaining_vertices = [v for v in adj if adj[v]]
    while remaining_vertices:
        start = remaining_vertices[0]
        stack = [start]
        chain = []
        
        while stack:
            curr = stack[-1]
            if adj[curr]:
                next_node = adj[curr].pop()
                if next_node in adj and curr in adj[next_node]:
                    adj[next_node].remove(curr)
                
                edge_key = tuple(sorted([curr, next_node]))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    stack.append(next_node)
            else:
                chain.append(stack.pop())
        
        if len(chain) > 1:
            coord_chain = [vertices[i] for i in chain]
            chains.append(coord_chain)
        
        remaining_vertices = [v for v in adj if adj[v]]
    
    return chains

def _filter_short_chains(vertices, edges, min_chain_len, min_segments):
    """
    Buduje łańcuchy (chain_segments) i odrzuca te krótsze niż min_chain_len
    lub z mniejszą liczbą segmentów niż min_segments. Zwraca listę linii.
    """
    if not edges:
        return []
    chains = chain_segments(edges, vertices)
    out_lines = []
    for chain in chains:
        segs = len(chain) - 1
        if segs < min_segments:
            continue
        total = 0.0
        for i in range(segs):
            total += np.linalg.norm(chain[i+1] - chain[i])
        if total >= min_chain_len:
            for i in range(segs):
                out_lines.append((chain[i].copy(), chain[i+1].copy()))
    return out_lines

def remove_artifacts(lines, min_seg_len=None, min_chain_len=None, min_segments=None):
    """
    Usuwa krótkie pojedyncze segmenty oraz krótkie łańcuchy.
    Zwraca przefiltrowaną listę linii.
    """
    if not lines:
        return []
    if min_seg_len is None:
        min_seg_len = Config.MIN_SEGMENT_LENGTH
    if min_chain_len is None:
        min_chain_len = Config.MIN_CHAIN_LENGTH
    if min_segments is None:
        min_segments = Config.MIN_SEGMENTS_IN_CHAIN

    vertices, edges = _lines_to_graph(lines, quant=4)
    # Usuń wiszące krótkie odcinki
    pruned = _prune_short_dangles(vertices, edges, min_seg_len)
    # Filtruj krótkie łańcuchy
    clean = _filter_short_chains(vertices, pruned, min_chain_len, min_segments)
    if clean:
        return clean
    # Fallback: usuń tylko krótkie pojedyncze segmenty
    return [ln for ln in lines if np.linalg.norm(ln[1] - ln[0]) >= min_seg_len]

def optimize_chain_order(chains):
    """Lekka heurystyka minimalizująca przeskoki między łańcuchami."""
    if not chains or len(chains) <= 1:
        return chains

    ordered = [chains.pop(0)]

    while chains:
        last_pt = np.asarray(ordered[-1][-1])
        best_i = None
        best_d = 1e20
        reverse = False

        for i, c in enumerate(chains):
            c0 = np.asarray(c[0])
            c1 = np.asarray(c[-1])
            d0 = np.linalg.norm(last_pt - c0)
            d1 = np.linalg.norm(last_pt - c1)

            if d0 < best_d:
                best_d = d0
                best_i = i
                reverse = False
            if d1 < best_d:
                best_d = d1
                best_i = i
                reverse = True

        chain = chains.pop(best_i)
        if reverse:
            chain = chain[::-1]
        ordered.append(chain)

    return ordered

# ---------------------
# Generowanie linii - ULEPSZONE (szybsza wersja)
# ---------------------
@measure_time
def generate_lines_from_mesh(mesh, camera_pos, threshold=0.1, wireframe=False):
    """Szybsza, stabilna wersja outline/wireframe."""

    vertices = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)

    if tri.shape[0] == 0:
        return []

    normals = np.asarray(mesh.triangle_normals)
    centers = vertices[tri].mean(axis=1)

    # BACKFACE CULLING (wrapper wywoła compute_backface_culling_numba gdy numba dostępna)
    visible = backface_culling(centers, normals, camera_pos, threshold)
    if not np.any(visible):
        return []

    tri = tri[visible]

    # GENEROWANIE KRAWĘDZI + klucze — zoptymalizowane dla bardzo dużych siatek
    num_tri = len(tri)
    if num_tri > LARGE_MESH_TRI and not NUMBA_AVAILABLE:
        # wielowątkowe budowanie kluczy na kawałkach (ThreadPool, bo numpy wektoryzuje pracę)
        n_workers = min(8, (os.cpu_count() or 4))
        # dzielimy na równomierne chunki
        chunk_size = (num_tri + n_workers - 1) // n_workers
        chunks = [tri[i*chunk_size:(i+1)*chunk_size] for i in range(n_workers) if i*chunk_size < num_tri]

        keys_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as exe:
            futures = [exe.submit(_edge_keys_from_tri_chunk, c) for c in chunks]
            for f in concurrent.futures.as_completed(futures):
                keys_list.append(f.result())

        if keys_list:
            all_keys = np.concatenate(keys_list)
        else:
            all_keys = np.empty(0, dtype=np.uint64)
    else:
        # standardowy szybki wektorowy wariant (jednowątkowy lub numba later)
        e0 = tri[:, [0, 1]]
        e1 = tri[:, [1, 2]]
        e2 = tri[:, [2, 0]]
        edges = np.vstack([e0, e1, e2])
        mins = np.minimum(edges[:, 0], edges[:, 1]).astype(np.uint64)
        maxs = np.maximum(edges[:, 0], edges[:, 1]).astype(np.uint64)
        all_keys = (mins << np.uint64(32)) | maxs

    if all_keys.size == 0:
        return []

    uniq_keys, counts = np.unique(all_keys, return_counts=True)

    # Wyciągamy tylko "jednostronne" krawędzie — czyli outline
    boundary_keys = uniq_keys[counts == 1]

    if len(boundary_keys) == 0:
        return []

    # Odtwarzamy pary vertexów
    boundary = np.column_stack([
        (boundary_keys >> 32).astype(np.int32),
        (boundary_keys & np.uint64(0xFFFFFFFF)).astype(np.int32)
    ])

    # ŁĄCZENIE W ŁAŃCUCHY
    chains = chain_segments(boundary.tolist(), vertices)
    if not chains:
        return [(vertices[a], vertices[b]) for a, b in boundary]

    chains = optimize_chain_order(chains)

    # KONWERSJA NA LINIE
    lines = []
    for chain in chains:
        for i in range(len(chain) - 1):
            lines.append((chain[i], chain[i + 1]))

    return remove_artifacts(lines)

@measure_time
def generate_lines(original_mesh, simplified_mesh, camera_pos, threshold=0.1, wireframe=False):
    """Generuje linie z obu meshy - oryginalnego i uproszczonego"""
    if wireframe:
        # Dla wireframe używamy tylko uproszczonego mesha
        return generate_lines_from_mesh(simplified_mesh, camera_pos, threshold, True)
    else:
        # Dla outline używamy OBIEGU meshy dla maksymalnej ciągłości
        original_lines = generate_lines_from_mesh(original_mesh, camera_pos, threshold, False)
        simplified_lines = generate_lines_from_mesh(simplified_mesh, camera_pos, threshold, False)
        
        # Połącz linie z obu meshy, preferując oryginalny dla lepszej jakości
        combined_lines = []
        
        # Najpierw dodaj linie z oryginalnego mesha (bardziej szczegółowe)
        combined_lines.extend(original_lines)
        
        # Dodaj unikalne linie z uproszczonego mesha, które uzupełniają luki
        original_set = set()
        for line in original_lines:
            # Tworzymy klucz niezależny od kolejności punktów
            key = tuple(sorted([tuple(line[0].round(4)), tuple(line[1].round(4))]))
            original_set.add(key)
        
        for line in simplified_lines:
            key = tuple(sorted([tuple(line[0].round(4)), tuple(line[1].round(4))]))
            if key not in original_set:
                combined_lines.append(line)
        
        # Usuń artefakty z połączonego zestawu
        combined_lines = remove_artifacts(combined_lines)
        
        return combined_lines

@measure_time
def optimize_path(lines, max_lines=5000):
    """Optymalizuje kolejność linii dla minimalnego czasu podróży"""
    if not lines or len(lines) <= 1:
        return lines
    
    # Ogranicz liczbę linii jeśli jest zbyt duża
    if len(lines) > max_lines:
        indices = np.linspace(0, len(lines)-1, max_lines, dtype=int)
        lines = [lines[i] for i in indices]
    
    # Konwertuj na numpy arrays dla wydajności
    starts = np.array([l[0] for l in lines], dtype=np.float32)
    ends = np.array([l[1] for l in lines], dtype=np.float32)
    
    optimized = []
    used = np.zeros(len(lines), dtype=bool)
    
    # Zacznij od linii najbliżej środka
    center = np.mean(np.vstack([starts, ends]), axis=0)
    distances = np.linalg.norm(starts - center, axis=1)
    start_idx = np.argmin(distances)
    
    optimized.append(lines[start_idx])
    used[start_idx] = True
    current_end = ends[start_idx]
    
    # Optymalizuj kolejność pozostałych linii
    for _ in range(len(lines) - 1):
        if NUMBA_AVAILABLE:
            best_idx, flip = find_nearest_numba(current_end, starts, ends, used)
        else:
            unused_mask = ~used
            if not np.any(unused_mask):
                break
                
            unused_indices = np.where(unused_mask)[0]
            dist_to_starts = np.linalg.norm(starts[unused_indices] - current_end, axis=1)
            dist_to_ends = np.linalg.norm(ends[unused_indices] - current_end, axis=1)
            
            min_start_idx = np.argmin(dist_to_starts)
            min_end_idx = np.argmin(dist_to_ends)
            
            if dist_to_starts[min_start_idx] < dist_to_ends[min_end_idx]:
                best_idx = unused_indices[min_start_idx]
                flip = False
            else:
                best_idx = unused_indices[min_end_idx]
                flip = True
        
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
# Eksport i skalowanie - ZMODYFIKOWANE
# ---------------------
@measure_time
def load_and_scale_mesh(path, max_triangles=None):
    """Wczytuje i skaluje mesh – zoptymalizowana wersja (2–5× szybsza)."""
    mesh = o3d.io.read_triangle_mesh(path)

    # Zachowaj tylko kopię geometrii wierzchołków – nie całego mesha
    original_vertices = np.asarray(mesh.vertices).copy()
    original_triangles = np.asarray(mesh.triangles).copy()

    # Uproszczenie (duże przyspieszenie – nie liczymy nic podwójnie)
    if max_triangles and len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
        print(f"Decimation: {len(original_triangles)} → {len(mesh.triangles)}")

    # Wyczyszczenie tylko tego, co niezbędne
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()

    # --- Skalowanie ---
    paper_w, paper_h = Config.PAPER_SIZES[Config.PAPER_FORMAT]
    if Config.PAPER_LANDSCAPE:
        paper_w, paper_h = paper_h, paper_w

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    center = np.asarray(bbox.get_center())

    usable = np.array([paper_w, paper_h]) - 2 * Config.PAPER_MARGIN
    max_z = Config.ROBOT_WORKSPACE_Z - Config.ROBOT_SAFE_Z - 10

    scale = min(
        usable[0] / max(extent[0], 1e-6),
        usable[1] / max(extent[1], 1e-6),
        max_z     / max(extent[2], 1e-6),
    )

    # macierz transformacji (szybciej niż osobne działania)
    T = np.eye(4)
    T[0, 0] = scale
    T[1, 1] = scale
    T[2, 2] = scale

    T[:3, 3] = [
        paper_w / 2 - center[0] * scale,
        paper_h / 2 - center[1] * scale,
        Config.ROBOT_SAFE_Z + (extent[2] * scale) / 2 - center[2] * scale
    ]

    mesh.transform(T)

    # Zastosuj tę samą transformację do oryginalnego mesha
    original = o3d.geometry.TriangleMesh()
    original.vertices = o3d.utility.Vector3dVector(original_vertices)
    original.triangles = o3d.utility.Vector3iVector(original_triangles)
    original.transform(T)
    original.compute_triangle_normals()

    print(f"Mesh scaled: scale={scale:.3f}, triangles={len(mesh.triangles)}")

    return original, mesh

# ---------------------
# Main - ZMODYFIKOWANE
# ---------------------
def main(mesh_path, camera_pos=None, look_at=None, dev_mode=False, save_mesh=False, logo_type=None):
    print("=" * 50)
    print("MESH -> ROBOT PATH CONVERTER")
    print("=" * 50)
    
    # Wczytaj OBIE wersje mesha - oryginalną i uproszczoną
    original_mesh, simplified_mesh = load_and_scale_mesh(mesh_path, Config.MAX_TRIANGLES)
    bounds = simplified_mesh.get_axis_aligned_bounding_box()
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
            # Outline: zapisz osobno oryginalny, uproszczony i merged
            orig_lines = generate_lines_from_mesh(original_mesh, cam, Config.VISIBILITY_THRESHOLD, wireframe=False)
            simpl_lines = generate_lines_from_mesh(simplified_mesh, cam, Config.VISIBILITY_THRESHOLD, wireframe=False)
            merged_lines = generate_lines(original_mesh, simplified_mesh, cam, Config.VISIBILITY_THRESHOLD, wireframe=False)

            # NOWE: orig_optimized bazujące na oryginalnym meshu (uproszczone łańcuchy)
            orig_optimized_lines = generate_orig_optimized(original_mesh, cam, view_dir, Config.VISIBILITY_THRESHOLD, epsilon=0.5, min_noise=0.05, area=0.01)

            if orig_lines:
                orig_opt = optimize_path(orig_lines)
                render_png(orig_opt, os.path.join(png_dir, f"{name}_outline_orig.png"), view_dir)
                render_svg(orig_opt, os.path.join(svg_dir, f"{name}_outline_orig.svg"), view_dir, logo_type)
            if simpl_lines:
                simpl_opt = optimize_path(simpl_lines)
                render_png(simpl_opt, os.path.join(png_dir, f"{name}_outline_simpl.png"), view_dir)
                render_svg(simpl_opt, os.path.join(svg_dir, f"{name}_outline_simpl.svg"), view_dir, logo_type)
            if merged_lines:
                merged_opt = optimize_path(merged_lines)
                render_png(merged_opt, os.path.join(png_dir, f"{name}_outline.png"), view_dir)
                render_svg(merged_opt, os.path.join(svg_dir, f"{name}_outline.svg"), view_dir, logo_type)

            # NOWE: zapisz orig_optimized
            if orig_optimized_lines:
                oo_opt = optimize_path(orig_optimized_lines)
                render_png(oo_opt, os.path.join(png_dir, f"{name}_outline_orig_optimized.png"), view_dir)
                render_svg(oo_opt, os.path.join(svg_dir, f"{name}_outline_orig_optimized.svg"), view_dir, logo_type)

            # Wireframe (jak wcześniej) - tylko uproszczony mesh
            wf_lines = generate_lines_from_mesh(simplified_mesh, cam, Config.VISIBILITY_THRESHOLD, wireframe=True)
            if wf_lines:
                wf_opt = optimize_path(wf_lines)
                render_png(wf_opt, os.path.join(png_dir, f"{name}_wireframe.png"), view_dir)
                render_svg(wf_opt, os.path.join(svg_dir, f"{name}_wireframe.svg"), view_dir, logo_type)
    else:
        if camera_pos is None:
            dist = max(extent[0], extent[1]) * 2.5
            camera_pos = center + np.array([0, 0, dist])
            look_at = center
            R = simplified_mesh.get_rotation_matrix_from_xyz((0, 0, np.pi))
            simplified_mesh.rotate(R, center=center)
            original_mesh.rotate(R, center=center)
        
        view_dir = (look_at - camera_pos) / np.linalg.norm(look_at - camera_pos)
        
        # Outline: oryginalny, uproszczony i merged
        orig_lines = generate_lines_from_mesh(original_mesh, camera_pos, Config.VISIBILITY_THRESHOLD, wireframe=False)
        simpl_lines = generate_lines_from_mesh(simplified_mesh, camera_pos, Config.VISIBILITY_THRESHOLD, wireframe=False)
        merged_lines = generate_lines(original_mesh, simplified_mesh, camera_pos, Config.VISIBILITY_THRESHOLD, wireframe=False)

        # NOWE: orig_optimized
        orig_optimized_lines = generate_orig_optimized(original_mesh, camera_pos, view_dir, Config.VISIBILITY_THRESHOLD, epsilon=0.5, min_noise=0.05, area=0.01)

        if orig_lines:
            orig_opt = optimize_path(orig_lines)
            render_png(orig_opt, os.path.join(Config.OUTPUT_DIR, "outline_orig.png"), view_dir)
            render_svg(orig_opt, os.path.join(Config.OUTPUT_DIR, "outline_orig.svg"), view_dir, logo_type)
        if simpl_lines:
            simpl_opt = optimize_path(simpl_lines)
            render_png(simpl_opt, os.path.join(Config.OUTPUT_DIR, "outline_simpl.png"), view_dir)
            render_svg(simpl_opt, os.path.join(Config.OUTPUT_DIR, "outline_simpl.svg"), view_dir, logo_type)
        if merged_lines:
            merged_opt = optimize_path(merged_lines)
            render_png(merged_opt, os.path.join(Config.OUTPUT_DIR, "outline.png"), view_dir)
            render_svg(merged_opt, os.path.join(Config.OUTPUT_DIR, "outline.svg"), view_dir, logo_type)
        
        # NOWE: zapisz orig_optimized (poza resztą)
        if orig_optimized_lines:
            oo_opt = optimize_path(orig_optimized_lines)
            render_png(oo_opt, os.path.join(Config.OUTPUT_DIR, "outline_orig_optimized.png"), view_dir)
            render_svg(oo_opt, os.path.join(Config.OUTPUT_DIR, "outline_orig_optimized.svg"), view_dir, logo_type)

        # Wireframe (jak wcześniej) - tylko uproszczony mesh
        wf_lines = generate_lines_from_mesh(simplified_mesh, camera_pos, Config.VISIBILITY_THRESHOLD, wireframe=True)
        if wf_lines:
            wf_opt = optimize_path(wf_lines)
            render_png(wf_opt, os.path.join(Config.OUTPUT_DIR, "wireframe.png"), view_dir)
            render_svg(wf_opt, os.path.join(Config.OUTPUT_DIR, "wireframe.svg"), view_dir, logo_type)
    
    if save_mesh:
        o3d.io.write_triangle_mesh(os.path.join(Config.OUTPUT_DIR, "mesh_original.ply"), original_mesh)
        o3d.io.write_triangle_mesh(os.path.join(Config.OUTPUT_DIR, "mesh_simplified.ply"), simplified_mesh)
        print(f"Meshe zapisane: {Config.OUTPUT_DIR}/mesh_*.ply")
    
    print(f"\n✅ Gotowe! -> {Config.OUTPUT_DIR}/")

def _rdp_indices(points, eps):
    """Zwraca indeksy punktów po uproszczeniu RDP (2D)"""
    if len(points) <= 2:
        return list(range(len(points)))

    def point_line_distance(pt, a, b):
        # odległość punktu pt od odcinka ab (2D)
        if np.allclose(a, b):
            return np.linalg.norm(pt - a)
        num = abs((b[0]-a[0])*(a[1]-pt[1]) - (a[0]-pt[0])*(b[1]-a[1]))
        den = np.hypot(b[0]-a[0], b[1]-a[1])
        return num / den

    idxs = []

    def rdp_rec(s, e):
        if e <= s + 1:
            return
        a = points[s]; b = points[e]
        max_d = -1.0; max_i = -1
        for i in range(s+1, e):
            d = point_line_distance(points[i], a, b)
            if d > max_d:
                max_d = d; max_i = i
        if max_d > eps:
            rdp_rec(s, max_i)
            rdp_rec(max_i, e)
        else:
            # no intermediate points needed
            idxs.append(s)
            idxs.append(e)

    rdp_rec(0, len(points)-1)
    if not idxs:
        return list(range(len(points)))
    # uporządkuj i usuń duplikaty
    idxs = sorted(set(idxs))
    # upewnij się, że są zawarte końce
    if idxs[0] != 0:
        idxs.insert(0, 0)
    if idxs[-1] != len(points)-1:
        idxs.append(len(points)-1)
    return idxs

def generate_orig_optimized(original_mesh, camera_pos, view_dir, threshold=0.1, epsilon=0.5, min_noise=0.05, area=0.01):
    """
    Wyciąga outline z ORYGINALNEGO mesha i upraszcza łańcuchy (RDP + filtracja)
    Zwraca listę linii [(p0,p1), ...] gdzie p* to 3D numpy array.
    """
    # Wczesne warunki
    if len(original_mesh.triangles) == 0:
        return []

    vertices = np.asarray(original_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(original_mesh.triangles, dtype=np.int32)

    if len(original_mesh.triangle_normals) == 0:
        original_mesh.compute_triangle_normals()
    normals = np.asarray(original_mesh.triangle_normals, dtype=np.float64)
    centers = np.mean(vertices[triangles], axis=1)

    visible_mask = backface_culling(centers, normals, camera_pos, threshold)
    visible_tris = triangles[visible_mask]

    if len(visible_tris) == 0:
        return []

    # policz krawędzie i wybierz te graniczne
    edge_counter = Counter()
    for tri in visible_tris:
        for i in range(3):
            edge = tuple(sorted([int(tri[i]), int(tri[(i+1)%3])]))
            edge_counter[edge] += 1
    boundary_edges = [e for e, c in edge_counter.items() if c == 1]
    if not boundary_edges:
        return []

    # połącz krawędzie w łańcuchy (zwraca listę list punktów 3D)
    chains = chain_segments(boundary_edges, vertices)
    if not chains:
        return []

    axes = get_projection_axes(view_dir)
    simplified_chains = []

    for chain in chains:
        pts3 = np.array(chain)  # (N,3)
        pts2 = pts3[:, axes]    # projekcja 2D

        # filtracja małych elementów po powierzchni w rzucie 2D
        bbox = pts2.max(axis=0) - pts2.min(axis=0)
        if bbox[0] * bbox[1] < area:
            continue

        # uproszczenie RDP (zwracamy indeksy w oryginalnym łańcuchu)
        idxs = _rdp_indices(pts2, epsilon)
        if len(idxs) < 2:
            continue

        # policz długość uproszczonego łańcucha i odrzuć za krótki (szum)
        total_len = 0.0
        for i in range(len(idxs)-1):
            total_len += np.linalg.norm(pts3[idxs[i+1]] - pts3[idxs[i]])
        if total_len < min_noise:
            continue

        simp = [pts3[i].copy() for i in idxs]
        # minimalnie: jeśli RDP zwróci zbyt mało punktów, zachowaj oryginalne
        if len(simp) >= 2:
            simplified_chains.append(simp)

    if not simplified_chains:
        return []

    # konwersja łańcuchów na listę linii
    lines = []
    for ch in simplified_chains:
        for i in range(len(ch)-1):
            lines.append((ch[i], ch[i+1]))

    # usuń artefakty (krótkie segmenty/łańcuchy)
    lines = remove_artifacts(lines)
    return lines

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