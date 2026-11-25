# Mesh to 2D Robot Path Converter (SVG)

Konwertuje pliki mesh (PLY, OBJ, STL) na ścieżki dla robota rysującego w postaci plików svg.

## Użycie

```bash 
uv sync
uv run main.py ./model.ply
```

### Opcje

```bash
uv run main.py model.ply --camera 100 100 200 --target 100 100 50
```

## Wyjście

- `output/render.png` - podgląd wireframe
- `output/output.svg` - plik SVG do rysowania
