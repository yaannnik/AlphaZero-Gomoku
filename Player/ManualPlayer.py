"""
@author: yangyi
"""

class ManualPlayer:
    def __init__(self):
        self.id = 0

    def set_id(self, p):
        self.id = p

    def get_action(self, cb):
        while True:
            pos = input("Make your move (row, col): ")
            if pos == '':
                continue
            pos = pos.split(",")
            row, col = int(pos[0]), int(pos[1])
            move = row * cb.size + col

            if 0 <= row < cb.size and 0 <= col < cb.size and move in cb.vacants:
                return move

            print("Invalid move, please try again!")

    def __str__(self):
        return "Manual Player %d" % self.id
