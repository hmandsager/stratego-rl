# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

- `npm run dev` — Start Vite dev server (port 5173)
- `npm run build` — TypeScript type-check + Vite production build
- `npm run lint` — Run ESLint
- `npm run preview` — Preview production build

No test framework is currently configured.

## Tech Stack

React 18 + TypeScript (strict mode) + Vite + Tailwind CSS 3. Icons from lucide-react. ESLint with TypeScript ESLint and React Hooks plugins.

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
