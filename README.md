# Aptitude Test Practice & Solver

Free, offline practice tools for psychometric tests used in job recruitment, plus an automated solver for the **Deductive (Grid)** coding test.

---

## 1. Practice Tools (Manual)
Download and open these HTML files in your browser to practice manually. No sign-up or internet connection required.

### `deductive-test.html` — Deductive (Figure Matrix)
A sudoku-style logic puzzle where each row and column must contain each symbol exactly once.
- **Grid sizes:** 4×4 (easy), 5×5 (medium/hard).
- **Features:** 6-minute timer, multiple difficulty levels, and session statistics (streaks, avg time).

### `inductive-test.html` — Inductive (Rule Finding)
Find the underlying rule for two categories of diamond-shaped cards, then classify new cards.
- **Rules:** Numeric patterns, letter types (vowels/consonants), and positional rules.
- **Features:** 20 questions, 12-minute timer, and detailed result breakdown.

---

## 2. Deductive Test Solver (Automated)
Located in the `/solver` directory, this Python script is specifically designed to solve the **Deductive/Coding** style test through real-time screen analysis. 

*Note: This solver is currently only compatible with the Deductive/Coding test format.*

### How it Works
* **Anchor Detection:** Locates the teal-colored "pipe" on the screen to establish coordinates.
* **Visual Analysis:** Uses OpenCV (HSV thresholding and contour analysis) to identify symbols like `red_square` or `green_triangle`.
* **OCR:** Utilizes `easyocr` to read numeric permutation codes from the screen.
* **Logic Engine:** Applies detected permutations to the top row and maps them to the bottom row to generate the final answer.

### Demo
<video src="solver/video/video.mp4" width="600" controls>
  Your browser does not support the video tag.
</video>

*Watch the solver in action: [Watch the demo video](solver/video/video.mp4)*

### Solver Installation & Usage
```bash
# Install dependencies
pip install opencv-python numpy pillow pyautogui easyocr

# Run the solver
python solver/solver_cv.py