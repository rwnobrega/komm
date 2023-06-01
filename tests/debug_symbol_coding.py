import komm

code = komm.SymbolCode({0: [1], 1: [0, 1], 2: [0, 0, 0], 3: [0, 0, 1]})
symbols = [1, 0, 1, 0, 2, 0]
bits = code.encode(symbols)
symbols_hat = code.decode(bits)
print(code)
print(symbols, bits, symbols_hat)
print(code.average_length([1 / 2, 1 / 4, 1 / 8, 1 / 8]))
