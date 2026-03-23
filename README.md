# Tetris AI Project

A high-performance Tetris Pygame and AI Tetris agent. This project implements a modular systems architecture to handle grid logic, block collisions, and heuristic-based move planning.

## Features
* Custom Game Engine: Built from scratch using Pygame.
* Heuristic AI: Real-time move evaluation and pathfinding.
* Modular Architecture: Logic decoupled into specialized modules for grid management, positioning, and movement execution.

## Dependencies
This project is built with Python 3.10+ and utilizes the following libraries:
* Pygame: Handles the graphical rendering and event loop.
* Copy: Standard library used for state-space deep copies during AI planning.
* Random: Manages randomized block generation logic.

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/tetris-ai-project.git
   cd tetris-ai-project

2. Set up a virtual environment:
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

## Usage

Run with the run.py in the root directory:
python3 run.py

## Libraries used

numpy
pandas
matplotlib
torch