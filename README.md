# Aptitude Test Practice

Free, offline practice tools for psychometric tests used in job recruitment. No sign-up, no subscription — just download and open in your browser.

## Tests included

### `deductive-test.html` — Deductive (Figure Matrix)
A sudoku-style logic puzzle where each row and column must contain each symbol exactly once.

- Grid sizes: 4×4 (easy), 5×5 (medium/hard)
- 6-minute timer per session
- All symbols shown — you decide what's missing
- Difficulty levels: Easy, Medium, Hard, Mix
- Session stats: solved/skipped, best time, streak, avg time per difficulty

### `inductive-test.html` — Inductive (Rule Finding)
Six diamond-shaped cards split into two categories. Find the rule, then classify four new cards.

- Diamond cards with 3×3 grids of letters and numbers
- Green category: filled circle bottom-left · empty circle top-right
- Gray category: filled circle top-right · empty circle bottom-left
- 20 questions, 12-minute timer
- Rule types: numeric patterns, letter patterns (vowels/consonants), positional rules, and more
- Results include: breakdown by difficulty, weakest rule types, avg response time per question, best streak, and full review of wrong answers

## How to use

```bash
git clone https://github.com/YOUR_USERNAME/aptitude-test-practice.git
cd aptitude-test-practice
```

Then open either HTML file directly in your browser — no server needed.

```
open deductive-test.html    # macOS
start deductive-test.html   # Windows
xdg-open deductive-test.html  # Linux
```

## Who is this for?

Anyone preparing for psychometric or logical reasoning tests used in job recruitment. These types of tests are common at large employers in finance, consulting, engineering, and the public sector.

## Tips for the real test

- **Deductive:** You typically have around 6 minutes for as many puzzles as possible — pace yourself, don't get stuck on one grid.
- **Inductive:** Look for the simplest rule first — it's usually about what type of content is in the grid (only digits, only vowels, etc.) before looking at position or sums.
- Practice daily in the week before your test. Even a few sessions significantly improves speed and pattern recognition.

## License

MIT — free to use, share, and modify.
