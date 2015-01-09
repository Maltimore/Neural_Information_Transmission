# This is the code for the neuron object! Not so much here yet :)

class Neuron:
    'This is a model neuron class that takes input and gives output'    
    
    def __init__(self, projections):
        # setting class parameters
        self.projections = projections
        
    def get_projections(self):
        return "My projections are: " + str(self.projections)