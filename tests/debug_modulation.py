import numpy as np
import komm


mod = komm.PAModulation(order=2)
#mod = komm.ASKModulation(order=4)
#mod = komm.PSKModulation(order=2)
#mod = komm.PSKModulation(order=4)
#mod = komm.PSKModulation(order=8)
#mod = komm.QAModulation(order=16, labeling='reflected_2d')
#mod = komm.ComplexModulation(constellation=[0, 1, 1j, -1])

M = mod.order
snr = 100.0
awgn = komm.AWGNChannel(snr=snr, signal_power=mod.energy_per_symbol)
mod.channel_snr = snr
num_symbols = 100_000
bits = np.random.randint(2, size=num_symbols * mod.bits_per_symbol)
sent = mod.modulate(bits)
recv = awgn(sent)
recv = [0.5 + 0.707j, -0.9 - 0.4j]
if isinstance(mod, komm.RealModulation):
    recv = np.real(recv)
demodulated_hard = mod.demodulate(recv, 'hard')
demodulated_soft = mod.demodulate(recv, 'soft')

print(f'mod = {mod}')
print(f'mod.constellation = {mod.constellation.tolist()}')
print(f'mod.order = {mod.order}')
print(f'mod.bits_per_symbol = {mod.bits_per_symbol}')
print(f'mod.labeling = {mod.labeling}')
print(f'mod._mapping = {mod._inverse_mapping}')
print(f'mod._inverse_mapping = {mod._inverse_mapping}')
print(f'mod.energy_per_symbol = {mod.energy_per_symbol}')
print(f'mod.energy_per_bit = {mod.energy_per_bit}')

print('---')
print(bits)
print(sent)
print(recv)
print(demodulated_hard)
print(demodulated_soft)
print(np.all(demodulated_hard == (demodulated_soft < 0)))
