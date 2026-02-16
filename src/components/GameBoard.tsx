import React from 'react';
import { Shield } from 'lucide-react';
import { BoardCell, Position, GameState } from '../types/game';
import { BOARD_SIZE } from '../utils/gameLogic';

interface GameBoardProps {
  gameState: GameState;
  onCellClick: (position: Position) => void;
}

export function GameBoard({ gameState, onCellClick }: GameBoardProps) {
  const { board, selectedPiece, currentTurn } = gameState;

  const renderPiece = (cell: BoardCell, pos: Position) => {
    if (!cell.piece) return null;

    const isSelected = selectedPiece?.row === pos.row && selectedPiece?.col === pos.col;
    const isPlayerPiece = cell.piece.player === 'red';

    return (
      <div
        className={`
          w-full h-full flex items-center justify-center
          ${isSelected ? 'bg-yellow-200' : ''}
          ${cell.piece.player === 'red' ? 'text-red-600' : 'text-blue-600'}
        `}
      >
        <Shield
          className={`w-8 h-8 ${isPlayerPiece || cell.piece.revealed ? '' : 'opacity-50'}`}
        />
        {(isPlayerPiece || cell.piece.revealed) && (
          <span className="absolute font-bold">
            {cell.piece.rank}
          </span>
        )}
      </div>
    );
  };

  return (
    <div className="grid grid-cols-10 gap-1 bg-gray-200 p-4 rounded-lg shadow-lg">
      {Array(BOARD_SIZE).fill(null).map((_, row) => (
        Array(BOARD_SIZE).fill(null).map((_, col) => {
          const cell = board[row][col];
          const isValidMove = selectedPiece && !cell.isLake &&
            (!cell.piece || cell.piece.player !== currentTurn);

          return (
            <div
              key={`${row}-${col}`}
              onClick={() => onCellClick({ row, col })}
              className={`
                w-12 h-12 relative cursor-pointer
                ${cell.isLake ? 'bg-blue-300' : 'bg-green-100'}
                ${isValidMove ? 'bg-yellow-100' : ''}
                hover:bg-opacity-80 transition-colors
                border-2 ${cell.isLake ? 'border-blue-400' : 'border-green-200'}
              `}
            >
              {renderPiece(cell, { row, col })}
            </div>
          );
        })
      ))}
    </div>
  );
}