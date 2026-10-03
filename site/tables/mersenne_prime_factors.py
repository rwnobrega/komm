import json
import os

from sympy import factorint, isprime

output_file = "mersenne_prime_factors.json"
max_k = 128

if not os.path.exists(output_file):
    table = {}
    for k in range(1, max_k + 1):
        print(k)
        factors = factorint(2**k - 1)
        assert all(isprime(p) for p in factors)
        table[k] = sorted(p for p, e in factors.items() for _ in range(e))

    # One line per k.
    rows = [f'  "{k}": {json.dumps(primes)}' for k, primes in table.items()]
    open(output_file, "w").write("{\n" + ",\n".join(rows) + "\n}\n")

table = json.load(open(output_file, "r"))
for k, primes in table.items():
    print(f"        {k}: {primes},")
