# AI Snake Game

## Project Overview
This project is an implementation of the classic Snake game enhanced with AI capabilities. The AI uses reinforcement learning techniques to play the game autonomously, making decisions based on the current state of the environment. The project includes multiple versions of the Snake game and policies for training and testing the AI.

## Features
- Classic Snake game implementation.
- AI-powered gameplay using reinforcement learning.
- Multiple versions of the Snake game for experimentation.
- Visualization of the game environment.
- Ability to create GIFs or videos of gameplay.

## Installation

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.9 or later
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/AI-Snake.git
   cd AI-Snake
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Running the Snake Game
1. Open a terminal and navigate to the project directory.
2. Run the desired version of the Snake game:
   ```bash
   python snake.py
   ```
   or
   ```bash
   python snake_v2.py
   ```
   or
   ```bash
   python snake_v3.py
   ```

### Running the AI
1. Open the Jupyter Notebook `SnakeAI.ipynb` in your preferred editor.
2. Execute the cells to train and test the AI.

### Creating GIFs or Videos
To create a GIF or video of the gameplay:
1. Use the `matplotlib.animation` module in Python.
2. Save the frames generated during gameplay as a GIF or video file.

## Project Structure
- `snake.py`: Base implementation of the Snake game.
- `snake_v2.py`: Enhanced version of the Snake game.
- `snake_v3.py`: Further improvements to the Snake game.
- `linear_policy.py`: Linear policy for AI decision-making.
- `log_policy.py`: Log-based policy for AI decision-making.
- `SnakeAI.ipynb`: Jupyter Notebook for training and testing the AI.
- `snake_game.gif`: Example GIF of gameplay.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## Contact
For any questions or feedback, please contact [your-email@example.com].
