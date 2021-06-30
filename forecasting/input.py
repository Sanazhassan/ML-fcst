class ModelInput:
    def __init__(self, location, part_number, parameters, var_names, train_data, train_var, hp, test_var=None):
        self.location = location
        self.part_number = part_number
        self.parameters = parameters
        self.var_names = var_names
        self.train_data = train_data
        self.train_var = train_var
        self.hp = hp
        self.test_var = test_var
