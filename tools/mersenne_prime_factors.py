import json
from pathlib import Path

from sympy import factorint, isprime

output_file = Path(__file__).with_suffix(".json")
max_k = 128

if not output_file.exists():
    table = {}
    for k in range(1, max_k + 1):
        print(k)
        factors = factorint(2**k - 1)
        assert all(isprime(p) for p in factors)
        table[k] = sorted(p for p, e in factors.items() for _ in range(e))

    # One line per k.
    rows = [f'  "{k}": {json.dumps(primes)}' for k, primes in table.items()]
    output_file.write_text("{\n" + ",\n".join(rows) + "\n}\n")

table = json.loads(output_file.read_text())
for k, primes in table.items():
    print(f"        {k}: {primes},")
