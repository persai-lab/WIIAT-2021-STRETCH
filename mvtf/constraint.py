class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            # print(module)
            w = module.weight.data
            w = w.clamp(0, 10)
            module.weight.data = w
