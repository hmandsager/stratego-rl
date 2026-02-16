import { Piece, Position, Rank, BoardCell } from '../types/game';

export const BOARD_SIZE = 10;

const PIECE_COUNTS: Record<Rank, number> = {
  'F': 1,  // Flag
  'B': 6,  // Bombs
  '1': 1,  // Marshal
  '2': 1,  // General
  '3': 2,  // Colonel
  '4': 3,  // Major
  '5': 4,  // Captain
  '6': 4,  // Lieutenant
  '7': 4,  // Sergeant
  '8': 5,  // Miner
  '9': 8,  // Scout
  'S': 1,  // Spy
};

export function createEmptyBoard(): BoardCell[][] {
  const board: BoardCell[][] = Array(BOARD_SIZE).fill(null).map(() =>
    Array(BOARD_SIZE).fill(null).map(() => ({ piece: null, isLake: false }))
  );

  // Set lake positions
  const lakePositions = [
    [4, 2], [4, 3], [5, 2], [5, 3],
    [4, 6], [4, 7], [5, 6], [5, 7]
  ];

  lakePositions.forEach(([row, col]) => {
    board[row][col].isLake = true;
  });

  return board;
}

export function initializeGame(): BoardCell[][] {
  const board = createEmptyBoard();
  const pieces = generateInitialPieces();
  
  // Place computer's pieces (blue)
  placePiecesRandomly(board, pieces.blue, 0, 3);
  
  // Place player's pieces (red)
  placePiecesRandomly(board, pieces.red, 6, 9);
  
  return board;
}

function generateInitialPieces(): { red: Piece[], blue: Piece[] } {
  const createPieces = (player: 'red' | 'blue') => {
    const pieces: Piece[] = [];
    Object.entries(PIECE_COUNTS).forEach(([rank, count]) => {
      for (let i = 0; i < count; i++) {
        pieces.push({
          id: `${player}-${rank}-${i}`,
          rank: rank as Rank,
          player,
          revealed: false
        });
      }
    });
    return pieces;
  };

  return {
    red: createPieces('red'),
    blue: createPieces('blue')
  };
}

function placePiecesRandomly(board: BoardCell[][], pieces: Piece[], startRow: number, endRow: number) {
  const availablePositions: Position[] = [];
  for (let row = startRow; row <= endRow; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      if (!board[row][col].isLake) {
        availablePositions.push({ row, col });
      }
    }
  }

  // Shuffle available positions
  for (let i = availablePositions.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [availablePositions[i], availablePositions[j]] = [availablePositions[j], availablePositions[i]];
  }

  pieces.forEach((piece, index) => {
    const pos = availablePositions[index];
    board[pos.row][pos.col].piece = piece;
  });
}

export function isValidMove(from: Position, to: Position, board: BoardCell[][]): boolean {
  const piece = board[from.row][from.col].piece;
  if (!piece) return false;

  // Bombs and Flags cannot move
  if (piece.rank === 'B' || piece.rank === 'F') return false;
  
  // Check if destination is a lake
  if (board[to.row][to.col].isLake) return false;
  
  // Check if destination has friendly piece
  const destPiece = board[to.row][to.col].piece;
  if (destPiece && destPiece.player === piece.player) return false;

  // Scout (rank 9) can move any number of empty spaces in a straight line
  if (piece.rank === '9') {
    if (from.row === to.row) {
      const start = Math.min(from.col, to.col);
      const end = Math.max(from.col, to.col);
      for (let col = start + 1; col < end; col++) {
        if (board[from.row][col].piece || board[from.row][col].isLake) return false;
      }
      return true;
    }
    if (from.col === to.col) {
      const start = Math.min(from.row, to.row);
      const end = Math.max(from.row, to.row);
      for (let row = start + 1; row < end; row++) {
        if (board[row][from.col].piece || board[row][from.col].isLake) return false;
      }
      return true;
    }
    return false;
  }

  // All other pieces can only move one space
  const rowDiff = Math.abs(to.row - from.row);
  const colDiff = Math.abs(to.col - from.col);
  return (rowDiff === 1 && colDiff === 0) || (rowDiff === 0 && colDiff === 1);
}

export function resolveCombat(attacker: Piece, defender: Piece): Piece | null {
  // Special cases
  if (defender.rank === 'F') return attacker; // Flag is captured
  if (defender.rank === 'B') {
    if (attacker.rank === '8') return attacker; // Miner defuses bomb
    return defender; // Bomb survives, attacker is destroyed
  }
  if (attacker.rank === 'S' && defender.rank === '1') return attacker; // Spy kills Marshal
  
  // Normal combat
  const attackerValue = parseInt(attacker.rank) || 0;
  const defenderValue = parseInt(defender.rank) || 0;
  
  if (attackerValue > defenderValue) return attacker;
  if (defenderValue > attackerValue) return defender;
  return null; // Both pieces are destroyed
}

export function executeMove(board: BoardCell[][], from: Position, to: Position): {
  board: BoardCell[][],
  gameOver: boolean,
  mover: 'red' | 'blue'
} {
  const newBoard: BoardCell[][] = JSON.parse(JSON.stringify(board));
  const movingPiece = newBoard[from.row][from.col].piece!;
  const targetCell = newBoard[to.row][to.col];

  if (targetCell.piece) {
    movingPiece.revealed = true;
    targetCell.piece.revealed = true;
    const winner = resolveCombat(movingPiece, targetCell.piece);

    if (winner === movingPiece) {
      targetCell.piece = movingPiece;
    } else if (winner === targetCell.piece) {
      // Defending piece wins, attacker is destroyed
    } else {
      targetCell.piece = null;
    }
  } else {
    targetCell.piece = movingPiece;
  }

  newBoard[from.row][from.col].piece = null;

  const opponentPlayer = movingPiece.player === 'red' ? 'blue' : 'red';
  const gameOver = !newBoard.flat().some(
    cell => cell.piece?.rank === 'F' && cell.piece.player === opponentPlayer
  );

  return { board: newBoard, gameOver, mover: movingPiece.player };
}

export function makeComputerMove(board: BoardCell[][]): { from: Position, to: Position } | null {
  const bluePieces: Position[] = [];
  const possibleMoves: { from: Position, to: Position }[] = [];

  // Find all blue pieces
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const piece = board[row][col].piece;
      if (piece && piece.player === 'blue') {
        bluePieces.push({ row, col });
      }
    }
  }

  // Find all possible moves
  bluePieces.forEach(from => {
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    directions.forEach(([dRow, dCol]) => {
      const to = { row: from.row + dRow, col: from.col + dCol };
      if (
        to.row >= 0 && to.row < BOARD_SIZE &&
        to.col >= 0 && to.col < BOARD_SIZE &&
        isValidMove(from, to, board)
      ) {
        possibleMoves.push({ from, to });
      }
    });
  });

  if (possibleMoves.length === 0) return null;

  // Simple AI: randomly choose a valid move
  const moveIndex = Math.floor(Math.random() * possibleMoves.length);
  return possibleMoves[moveIndex];
}