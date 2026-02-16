# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

### Frontend (React UI)
- `npm run dev` — Start Vite dev server (port 5173)
- `npm run build` — TypeScript type-check + Vite production build
- `npm run lint` — Run ESLint
- `npm run preview` — Preview production build

### Python (RL Training)
- `source venv/bin/activate` — Activate the Python virtual environment
- `uv pip install -r requirements.txt` — Install/update Python dependencies
- `uv pip install <package>` — Add a new package (then update requirements.txt)

All Python commands must be run with the `venv/` virtual environment activated or with `VIRTUAL_ENV=venv uv pip ...`. Use `uv` for all package management — do not use raw `pip`.

No test framework is currently configured.

## Tech Stack

**Frontend:** React 18 + TypeScript (strict mode) + Vite + Tailwind CSS 3. Icons from lucide-react. ESLint with TypeScript ESLint and React Hooks plugins.

**Python/RL:** Python 3.12 + PyTorch + Gymnasium + TensorBoard + NumPy + Matplotlib. Managed via `uv` with a local `venv/` virtual environment.

## Architecture

This is a Stratego board game where a human (red) plays against an AI opponent (blue).

**State management:** All game state lives in `App.tsx` as a single `GameState` object and flows down as props. No external state library.

**Game flow:** Player selects a piece then a destination → move is validated and executed → turn switches to AI → AI auto-moves after 1s delay (via `useEffect`) → game ends when a flag is captured.

**Key modules:**

- `src/types/game.ts` — Core type definitions: `Piece`, `Position`, `BoardCell`, `GameState`. Pieces have rank, player (`'red'|'blue'`), and revealed status.
- `src/utils/gameLogic.ts` — Pure functions for board initialization, move validation, combat resolution, and AI. Board is 10×10 with 8 fixed lake cells. Scouts move multiple spaces; all others move one. Combat uses rank comparison with special cases (Spy beats Marshal when attacking, Miner beats Bomb).
- `src/components/GameBoard.tsx` — Board rendering with click-based interaction. Color-coded cells (green=terrain, blue=lake, yellow=selected/valid moves). Enemy pieces are semi-transparent until revealed through combat.
- `src/App.tsx` — Orchestrates game state, handles cell clicks, executes moves, and triggers AI turns.

**AI:** Currently random move selection (`makeComputerMove`). The project name suggests reinforcement learning is the intended enhancement.

## Python Environment

The `venv/` directory contains the Python virtual environment created with `uv`. Key dependencies are listed in `requirements.txt`. Always use `uv` (not `pip`) for installing packages.
