import json
import os

from sympy import factorint, isprime

output_file = "mersenne_prime_factors.json"

if not os.path.exists(output_file):
    table = {}
    for k in range(1, 129):
        print(k)
        factors = factorint(2**k - 1)
        assert all(isprime(p) for p in factors)
        table[k] = sorted(p for p, e in factors.items() for _ in range(e))

    json.dump(table, open(output_file, "w"), indent=4)

table = json.load(open(output_file, "r"))
for k, primes in table.items():
    print(f"        {k}: {primes},")
