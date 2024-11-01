import inspect

import komm
from komm._error_control_block._registry import RegistryBlockDecoder

supported_decoders = {}
all_classes = []
for name, cls in inspect.getmembers(komm, inspect.isclass):
    if issubclass(cls, komm.BlockCode):
        supported_decoders[name] = cls.supported_decoders()
        all_classes.append(name)

for method in RegistryBlockDecoder.list():
    decoder_data = RegistryBlockDecoder.get(method)
    (description, type_in, type_out, target) = (
        decoder_data["description"],
        decoder_data["type_in"],
        decoder_data["type_out"],
        decoder_data["target"],
    )
    supported_by = [
        name for name, decoders in supported_decoders.items() if method in decoders
    ]

    if supported_by == all_classes:
        supported_by = "All codes"
    else:
        supported_by = ", ".join([f"[`{name}`](/ref/{name})" for name in supported_by])

    print(f"**`{method}`**: {description}")
    print()
    print(f"  - Input type: {type_in}")
    print(f"  - Output type: {type_out}")
    print(f"  - Target: {target}")
    print(f"  - Supported by: {supported_by}.")
    print()
