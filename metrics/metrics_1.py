class Metrics:
    def __init__(self):
        self.map = {}

    def add(self, key, start_time):
        if key in self.map:
            raise (Exception("metrics already added for {}".format(key)))
        self.map[key] = start_time

    def calculate(self, key, end_time):
        time_taken = end_time - self.map[key]
        self.map[key] = time_taken

    def print(self):
        for k, v in self.map.items():
            print("Time Taken = {}, {}".format(v, k))

    # key = "{}_{}_{}_{}".format(arg.group, arg.region, PY_ES, len(arg.train_data))
