class Greedy:
    def __init__(self):
        pass

    def action(self, a: int, b: int) -> int:
        return (0, a) if a > b else (1, b)
