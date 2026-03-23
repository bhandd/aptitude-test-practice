"""
AON Kodningstest Solver — OpenCV + färgdetektering
Ingen LLM behövs — ren bildanalys och mappningslogik.

Kör: python solver_cv.py
Kräver: pip install opencv-python numpy pillow pyautogui
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
import pyautogui
import time
from PIL import ImageGrab
import easyocr

# EasyOCR-läsare — laddas en gång vid start
_ocr_reader = None
def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        print("Laddar OCR-modell (en gång)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    return _ocr_reader

# ── Konfig ───────────────────────────────────────────────────
FPS      = 2      # Skärmavläsningar per sekund
COOLDOWN = 3.0    # Sekunder mellan lösningar
DEBUG    = True   # Spara debug-bilder (debug_*.png)

# ── Färgdefinitioner i HSV ───────────────────────────────────
# Format: (lower_hsv, upper_hsv)
# Justera om symbolerna på din skärm har lite avvikande färger

SYMBOL_COLORS = {
    "red":    (np.array([0,  120, 100]), np.array([10, 255, 255])),
    "orange": (np.array([11, 120, 100]), np.array([25, 255, 255])),
    "green":  (np.array([40, 100, 80]),  np.array([80, 255, 255])),
    "blue":   (np.array([100,120, 80]),  np.array([130,255, 255])),
    "purple": (np.array([130,80,  80]),  np.array([160,255, 255])),
    "pink":   (np.array([160,80,  80]),  np.array([179,255, 255])),
}

# Teal = rörets färg (ankarpunkt)
TEAL_LOWER = np.array([75, 100, 80])
TEAL_UPPER = np.array([95, 255, 255])

# Teal highlight = markerade koder ovanför röret
HIGHLIGHT_LOWER = np.array([75, 40, 180])
HIGHLIGHT_UPPER = np.array([95, 120, 255])

# ── Hjälpfunktioner ───────────────────────────────────────────

def grab_screen():
    img = np.array(ImageGrab.grab())
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def find_pipe(bgr):
    """Hitta rörets position via teal-färg. Returnerar (cx, cy, x, y, w, h) eller None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TEAL_LOWER, TEAL_UPPER)
    
    # Morfologi för att rensa brus
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Hitta alla teal-objekt och beräkna samlad bounding box
    all_points = np.concatenate([c for c in contours if cv2.contourArea(c) > 1000])
    if len(all_points) == 0:
        return None
    
    x, y, w, h = cv2.boundingRect(all_points)
    cx, cy = x + w // 2, y + h // 2
    print(f"    Rör bbox: x={x} y={y} w={w} h={h} cx={cx} cy={cy}")
    return cx, cy, x, y, w, h

def crop_zones(bgr, cx, cy, px, py, pw, ph):
    """
    Croppa ut de fyra zonerna relativt rörets faktiska position.
    px,py = rörboxens övre vänstra hörn, pw,ph = rörboxens bredd/höjd
    """
    H, W = bgr.shape[:2]
    half_w = 220
    sym_h  = 100  # höjd på symbolrad
    gap    = 15   # avstånd mellan rör och symbolrad
    
    # Zon-positioner relativt rörets centrum (cy)
    # Mätt från debug_full: rör cy≈415, top-symboler ca 130px ovan cy, bot ca 130px under
    # Använd cy (mitten av röret) som ankarpunkt
    top_y1    = cy - 200
    top_y2    = cy - 90
    bot_y1    = cy + 90
    bot_y2    = cy + 200
    hl_y1     = cy - 320
    hl_y2     = cy - 210

    zones = {
        "top":         bgr[max(0,top_y1):max(0,top_y2), max(0,cx-half_w):cx+half_w],
        "highlighted": bgr[max(0,hl_y1) :max(0,hl_y2),  max(0,cx-half_w):cx+half_w],
        "bottom":      bgr[max(0,bot_y1):max(0,bot_y2), max(0,cx-half_w):cx+half_w],
    }

    y_coords = {
        "top":         (top_y1+top_y2)//2,
        "highlighted": (hl_y1+hl_y2)//2,
        "bottom":      (bot_y1+bot_y2)//2,
    }
    
    return zones, y_coords

def detect_color(bgr_roi):
    """Returnerar den dominerande symbol-färgen i en ROI."""
    if bgr_roi is None or bgr_roi.size == 0:
        return "unknown"
    
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    best_color = "unknown"
    best_count = 50  # minsta antal pixlar för att räknas
    
    for color_name, (lower, upper) in SYMBOL_COLORS.items():
        mask = cv2.inRange(hsv, lower, upper)
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_color = color_name
    
    return best_color

def detect_shape(bgr_roi):
    """
    Detekterar form via konturanalys.
    Returnerar: 'circle', 'square', 'triangle', 'diamond', 'plus', 'star', 'unknown'
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return "unknown"
    
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Ta bort kanten
    thresh = cv2.erode(thresh, np.ones((3,3), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 100:
        return "unknown"
    
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return "unknown"
    
    # Cirkuläritet
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Approximera polygon
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    vertices = len(approx)
    
    # Aspect ratio för bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h > 0 else 1
    
    # Konvexitet (plus-tecken är konkavt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1
    
    if circularity > 0.80:
        return "circle"
    elif solidity < 0.70:
        return "plus"       # plus är konkavt (låg solidity)
    elif vertices == 3:
        return "triangle"
    elif vertices == 4:
        if 0.85 < aspect < 1.15:
            return "square"
        else:
            return "diamond"
    elif vertices >= 8 and circularity < 0.60:
        return "star"
    elif vertices == 4:
        return "diamond"
    else:
        return "unknown"

def split_symbol_row(zone_bgr, n=4):
    """
    Delar upp en symbolrad i n lika delar och returnerar varje del.
    """
    if zone_bgr is None or zone_bgr.size == 0:
        return [None] * n
    
    W = zone_bgr.shape[1]
    cell_w = W // n
    cells = []
    for i in range(n):
        x1 = i * cell_w
        x2 = (i+1) * cell_w
        cells.append(zone_bgr[:, x1:x2])
    return cells

def identify_symbol(cell_bgr):
    """Identifierar en symbol i en cell. Returnerar t.ex. 'red_square'."""
    color = detect_color(cell_bgr)
    shape = detect_shape(cell_bgr)
    return f"{color}_{shape}"

def find_highlighted_codes(bgr, cx, cy):
    """
    Letar efter teal-markerade koder ovanför tratten.
    Returnerar antal (0, 1 eller 2).
    Använder liten kernel så två koder nära varandra inte slås ihop.
    """
    y1 = max(0, cy - 300)
    y2 = cy - 30
    x1 = max(0, cx - 300)
    x2 = cx + 300
    
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HIGHLIGHT_LOWER, HIGHLIGHT_UPPER)
    
    # Liten kernel — inte slå ihop separata koder
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrera bort brus men acceptera mindre blobs
    significant = [c for c in contours if cv2.contourArea(c) > 300]
    # Sortera uppifrån ned
    significant.sort(key=lambda c: cv2.boundingRect(c)[1])
    
    if DEBUG:
        dbg = roi.copy()
        for c in significant:
            bx,by,bw,bh = cv2.boundingRect(c)
            cv2.rectangle(dbg,(bx,by),(bx+bw,by+bh),(0,255,0),2)
        cv2.imwrite("debug_highlight_detect.png", dbg)
    
    print(f"    Highlight-blobs: {len(significant)}")
    return min(len(significant), 2)

def read_code_from_zone(bgr_zone):
    """Läser en sifferkod (t.ex. '4 1 3 2') från en bildzon med EasyOCR."""
    if bgr_zone is None or bgr_zone.size == 0:
        return None
    try:
        # Förstora bilden för bättre OCR
        scale = 3
        big = cv2.resize(bgr_zone, (bgr_zone.shape[1]*scale, bgr_zone.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
        # Konvertera till RGB
        rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
        reader = get_ocr()
        results = reader.readtext(rgb, allowlist='1234 ', detail=0)
        # Samla alla siffror från resultaten
        digits = []
        for r in results:
            for c in r:
                if c in '1234':
                    digits.append(int(c))
        if len(digits) == 4:
            return digits
        # Om vi fick fler eller färre, försök ta första 4
        if len(digits) > 4:
            return digits[:4]
    except Exception as e:
        print(f"    OCR-fel: {e}")
    return None

# ── Lösningslogik ─────────────────────────────────────────────

def apply_perm(mapping, perm):
    """
    Tillämpar en permutation på en mappning.
    perm = [3,1,4,2] betyder: ny pos0=gamla pos2, ny pos1=gamla pos0, osv.
    mapping[i] = vilket original-index som nu är på position i
    """
    new_mapping = [mapping[p-1] for p in perm]
    return new_mapping

def solve_coding(top_symbols, bot_symbols, highlighted_count, perms):
    """
    Löser kodningstestet.
    
    top_symbols: ['red_square', 'green_triangle', 'blue_plus', 'orange_diamond']
    bot_symbols:  samma symboler i ny ordning
    highlighted_count: 0, 1 eller 2
    perms: lista med permutationer [[3,1,4,2], ...] (en per highlighted kod)
    
    Returnerar: koden som lista [3,2,4,1]
    """
    # Starta med grundmappning: position i → symbol top_symbols[i]
    # mapping[i] = original index för symbolen på position i
    mapping = list(range(4))  # [0,1,2,3]
    
    # Tillämpa permutationer i ordning
    for perm in perms:
        mapping = apply_perm(mapping, perm)
    
    # Nu: mapping[i] = vilket original-index som är på ny position i
    # Omvänd mappning: original_idx → ny position (= ny siffra, 1-baserad)
    orig_to_num = {}
    for new_pos, orig_idx in enumerate(mapping):
        orig_to_num[orig_idx] = new_pos + 1
    
    # Koda bot_symbols
    result = []
    for sym in bot_symbols:
        if sym in top_symbols:
            orig_idx = top_symbols.index(sym)
            result.append(orig_to_num[orig_idx])
        else:
            result.append(0)  # okänd symbol
    
    return result

# ── Huvud-analyzer ────────────────────────────────────────────

def analyze_and_solve(bgr):
    """Analyserar skärmen och returnerar lösningen."""
    
    # 1. Hitta röret
    pipe = find_pipe(bgr)
    if pipe is None:
        return None
    cx, cy, px, py, pw, ph = pipe
    print(f"  Rör hittat vid ({cx}, {cy}), storlek {pw}x{ph}")
    
    # 2. Croppa zoner
    zones, y_coords = crop_zones(bgr, cx, cy, px, py, pw, ph)
    
    if DEBUG:
        for name, zone in zones.items():
            if zone is not None and zone.size > 0:
                cv2.imwrite(f"debug_{name}.png", zone)
    
    # 3. Identifiera top-symboler
    top_cells = split_symbol_row(zones["top"], 4)
    top_symbols = [identify_symbol(c) for c in top_cells]
    print(f"  Top: {top_symbols}")
    
    # 4. Identifiera bot-symboler
    bot_cells = split_symbol_row(zones["bottom"], 4)
    bot_symbols = [identify_symbol(c) for c in bot_cells]
    print(f"  Bot: {bot_symbols}")
    
    # 5. Kolla highlighted koder
    n_highlighted = find_highlighted_codes(bgr, cx, cy)
    print(f"  Highlighted koder: {n_highlighted}")
    
    # 6. Läs permutationskoderna med OCR om det finns highlighted koder
    perms = []
    if n_highlighted > 0:
        print(f"  Läser {n_highlighted} permutationskod(er) med OCR...")
        # Croppa varje highlighted kod separat
        # Hitta blob-positioner dynamiskt
        y1 = max(0, cy - 300)
        y2 = cy - 30
        x1 = max(0, cx - 300)
        x2 = cx + 300
        roi = bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HIGHLIGHT_LOWER, HIGHLIGHT_UPPER)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in contours if cv2.contourArea(c) > 300]
        blobs.sort(key=lambda c: cv2.boundingRect(c)[1])  # uppifrån ned

        for i in range(n_highlighted):
            if i < len(blobs):
                # Använd faktisk blob-position
                bx, by, bw, bh = cv2.boundingRect(blobs[i])
                pad = 12
                ry1 = max(0, y1 + by - pad)
                ry2 = min(bgr.shape[0], y1 + by + bh + pad)
                rx1 = max(0, x1 + bx - pad)
                rx2 = min(bgr.shape[1], x1 + bx + bw + pad)
                hl_zone = bgr[ry1:ry2, rx1:rx2]
            else:
                # Fallback: fast offset
                slot_h = 70
                gap = 70 + slot_h * (n_highlighted - 1 - i)
                hl_zone = bgr[max(0,cy-gap-slot_h):max(0,cy-gap), max(0,cx-180):cx+180]
            
            if DEBUG:
                cv2.imwrite(f"debug_perm_{i}.png", hl_zone)
            perm = read_code_from_zone(hl_zone)
            if perm:
                print(f"    Perm {i+1}: {perm}")
                perms.append(perm)
            else:
                print(f"    Perm {i+1}: kunde inte läsas")

    # 7. Lös
    result = solve_coding(top_symbols, bot_symbols, n_highlighted, perms)
    code_str = " ".join(str(x) for x in result)
    level = n_highlighted + 1
    print(f"\n  ✓ SVAR (nivå {level}): {code_str}")
    return {"code": code_str, "cx": cx, "cy": cy}

# ── Förändringsdetektion ──────────────────────────────────────

def compute_hash(bgr):
    small = cv2.resize(bgr, (64, 36))
    return small.mean(axis=2).flatten()

def screen_changed(old_hash, new_hash, threshold=0.03):
    if old_hash is None:
        return True
    diff = np.abs(old_hash.astype(float) - new_hash.astype(float)).mean()
    return diff > threshold * 255

# ── Huvudloop ─────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════╗")
    print("║   AON Solver — OpenCV            ║")
    print("║   Ingen LLM, ingen API-nyckel    ║")
    print("╚══════════════════════════════════╝\n")
    print(f"✓ Skannar {FPS}x/sek | Ctrl+C för att avsluta")
    print("  Öppna kodningstestet i webbläsaren\n")
    
    try:
        while True:
            # Steg 1: vänta tills röret syns på skärmen
            print("Väntar på test — byt till webbläsarfönstret nu!", end="\r")
            time.sleep(3)  # Ge tid att byta fönster
            bgr = grab_screen()
            while find_pipe(bgr) is None:
                time.sleep(0.5)
                bgr = grab_screen()

            # Steg 2: rör hittat — analysera och lös
            print(f"\n[{time.strftime('%H:%M:%S')}] Rör hittat — analyserar...")
            result = analyze_and_solve(bgr)

            if result and DEBUG:
                debug = bgr.copy()
                cx, cy = result["cx"], result["cy"]
                cv2.circle(debug, (cx, cy), 10, (0, 255, 0), 3)
                cv2.imwrite("debug_full.png", debug)
                print("  Debug-bilder sparade (debug_*.png)")

            # Steg 3: vänta 5 sekunder innan nästa sökning
            print("  Väntar 5 sek innan nästa fråga...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nAvslutar.")

if __name__ == "__main__":
    main()
