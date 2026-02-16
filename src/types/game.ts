export type Rank = 
  | 'F' // Flag
  | 'B' // Bomb
  | '1' // Marshal
  | '2' // General
  | '3' // Colonel
  | '4' // Major
  | '5' // Captain
  | '6' // Lieutenant
  | '7' // Sergeant
  | '8' // Miner
  | '9' // Scout
  | 'S'; // Spy

export type Piece = {
  id: string;
  rank: Rank;
  player: 'red' | 'blue';
  revealed: boolean;
};

export type Position = {
  row: number;
  col: number;
};

export type BoardCell = {
  piece: Piece | null;
  isLake: boolean;
};

export type GameState = {
  board: BoardCell[][];
  selectedPiece: Position | null;
  currentTurn: 'red' | 'blue';
  gameOver: boolean;
  winner: 'red' | 'blue' | null;
};