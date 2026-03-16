# AI---Group-5---Project-1

Reversi setup

Prereqs:
- Python 3.10+ (We used Python 3.12)

1) Install dependencies
    From the project directory, run:

    macOS:
        1. python3 -m venv .venv
        2. source .venv/bin/activate
        3. pip install -r requirements.txt

    Windows:
        1. Type (Control + Shift + p)
        2. Select 'Python: Select Interpreter'
        3. Select 'Create Virtual Environment'
        4. Select 'venv'
        5. Select 'requirements.txt'
        6. Activate virtual environment by typing 'source .venv/Scripts/activate'

2) Run game
    Open 3 terminals in the project folder (with the same virtual environment activated using (source .venv/bin/activate) for mac)

    Terminal 1 (Reversi server):
    python3 reversi_server.py

    Terminal 2 (player 1 - White):
    python3 minimax_player.py

    Terminal 3 (player 2 - Black):
    python3 greedy_player.py

Notes:
    - After both players connect, click server window to start game
    - To force stop the code, run (control + c) in all 3 terminals
