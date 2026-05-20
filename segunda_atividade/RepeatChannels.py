class RepeatChannels:
    def __init__(self, n): self.n = n
    def __call__(self, x): return x.repeat(self.n, 1, 1)