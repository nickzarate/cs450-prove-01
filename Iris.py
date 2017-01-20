class Iris(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def test(self):
        print(self.data)
        print(self.target)