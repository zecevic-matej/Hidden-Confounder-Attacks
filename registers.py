# Paper published at ACML 2023 Journal Track
# Link to Conference: https://www.acml-conf.org/2023/papers.html
# corresponding author: Matej Zecevic, matej.zecevic@tu-darmstadt.de
class Register:
    def __init__(self):
        self.store = {}

    def register(self, key, function):
        self.store[key] = function
    
    def get(self, key):
        return self.store[key]


class FunctionRegister(Register):
    def __init__(self):
        super().__init__()


class NoiseRegister(Register):
    def __init__(self):
        super().__init__()


class VariableRegister(Register):
    def __init__(self):
        super().__init__()
        
class WeightsRegister(Register):
    def __init__(self):
        super().__init__()
