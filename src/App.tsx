import React, { useState, useEffect } from 'react';
import { GameBoard } from './components/GameBoard';
import { Position, GameState } from './types/game';
import { initializeGame, isValidMove, executeMove, makeComputerMove } from './utils/gameLogic';

function App() {
  const [gameState, setGameState] = useState<GameState>({
    board: initializeGame(),
    selectedPiece: null,
    currentTurn: 'red',
    gameOver: false,
    winner: null
  });

  useEffect(() => {
    if (gameState.currentTurn === 'blue' && !gameState.gameOver) {
      const timeoutId = setTimeout(() => {
        setGameState(prev => {
          if (prev.currentTurn !== 'blue' || prev.gameOver) return prev;

          const move = makeComputerMove(prev.board);
          if (!move) {
            return { ...prev, gameOver: true, winner: 'red' as const };
          }

          const result = executeMove(prev.board, move.from, move.to);
          return {
            ...prev,
            board: result.board,
            selectedPiece: null,
            currentTurn: 'red' as const,
            gameOver: result.gameOver,
            winner: result.gameOver ? 'blue' as const : null
          };
        });
      }, 1000);

      return () => clearTimeout(timeoutId);
    }
  }, [gameState.currentTurn, gameState.gameOver]);

  const handleCellClick = (position: Position) => {
    if (gameState.gameOver || gameState.currentTurn !== 'red') return;

    const { board, selectedPiece } = gameState;
    const clickedCell = board[position.row][position.col];

    // If no piece is selected and clicked on own piece, select it
    if (!selectedPiece && clickedCell.piece?.player === 'red') {
      setGameState(prev => ({ ...prev, selectedPiece: position }));
      return;
    }

    // If piece is selected and clicked on valid destination, move
    if (selectedPiece && isValidMove(selectedPiece, position, board)) {
      handleMove(selectedPiece, position);
    }

    // Deselect piece
    setGameState(prev => ({ ...prev, selectedPiece: null }));
  };

  const handleMove = (from: Position, to: Position) => {
    setGameState(prev => {
      const result = executeMove(prev.board, from, to);
      return {
        ...prev,
        board: result.board,
        selectedPiece: null,
        currentTurn: prev.currentTurn === 'red' ? 'blue' as const : 'red' as const,
        gameOver: result.gameOver,
        winner: result.gameOver ? result.mover : null
      };
    });
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold mb-8 text-gray-800">Stratego</h1>
      
      <div className="mb-4 text-lg">
        {gameState.gameOver ? (
          <div className="text-2xl font-bold text-center">
            Game Over! {gameState.winner === 'red' ? 'You' : 'Computer'} won!
          </div>
        ) : (
          <div className={`text-${gameState.currentTurn === 'red' ? 'red' : 'blue'}-600`}>
            {gameState.currentTurn === 'red' ? 'Your turn' : 'Computer is thinking...'}
          </div>
        )}
      </div>

      <GameBoard
        gameState={gameState}
        onCellClick={handleCellClick}
      />

      <div className="mt-8 text-gray-600">
        <h2 className="text-xl font-semibold mb-2">How to Play:</h2>
        <ul className="list-disc list-inside">
          <li>Click on your pieces (red) to select them</li>
          <li>Click on a valid destination to move</li>
          <li>Capture the enemy flag to win!</li>
          <li>Higher numbers defeat lower numbers</li>
          <li>Miners (8) can defuse Bombs (B)</li>
          <li>Spy (S) can defeat Marshal (1) if attacking</li>
        </ul>
      </div>
    </div>
  );
}

export default App;