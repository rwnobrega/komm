import komm

p = 0.125
transition_matrix = [[1-p, p], [p, 1-p]]
dmc = komm.DiscreteMemorylessChannel(transition_matrix)
print(dmc)
print(dmc.mutual_information([1/2, 1/2]))
