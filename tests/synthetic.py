"""Small, answer-known fixtures for offline pipeline tests (no visible data/key)."""

PEOPLE = ("Rhea Vale", "Kian", "Mara-June", "Noel", "Tao")
BLOCKS = ("07:00", "09:00", "11:00", "13:00", "15:00")
STATIONS = ("cold-store", "label room", "dispatch")
HOLDERS = PEOPLE[:3]

HEADER = (
    "Shift notes, Test Bay. 5 staff on rota: %s. Blocks run %s, one person per block, "
    "and each person works exactly one block. There are 3 stations, one person on each: %s. "
    "The people on a station are %s; the rest are on no station."
) % (", ".join(PEOPLE), ", ".join(BLOCKS), ", ".join(STATIONS), ", ".join(HOLDERS))

UNIQUE_BODY = [
    "Rhea Vale is on 07:00.",
    "Kian is on 09:00.",
    "Mara-June is on 11:00.",
    "Noel is on 13:00.",
    "Tao is on 15:00.",
    "Rhea Vale is on cold-store.",
    "Kian is on label room.",
    "Mara-June is on dispatch.",
]

BETWEEN_BODY = [
    "Rhea Vale works the 07:00 block.",
    "Kian works the 09:00 block.",
    "Tao works the 15:00 block.",
    "Mara-June is between Kian and Noel in the schedule, in either order.",
    "Rhea Vale holds cold-store.",
    "Kian holds label room.",
    "Mara-June holds dispatch.",
]


def item(item_id="SYNTH-000", body=None, header=HEADER):
    return {"id": item_id, "text": header + "\n" + "\n".join(body or UNIQUE_BODY)}


def items(count=60):
    return [item("SYNTH-%03d" % i) for i in range(count)]


def gold_for(items_to_label):
    """Synthetic, explicit records used only by fake-model plumbing tests."""
    readings = [
        ("AT", PEOPLE[0], BLOCKS[0]),
        ("AT", PEOPLE[1], BLOCKS[1]),
        ("AT", PEOPLE[2], BLOCKS[2]),
        ("AT", PEOPLE[3], BLOCKS[3]),
        ("AT", PEOPLE[4], BLOCKS[4]),
        ("ON", PEOPLE[0], STATIONS[0]),
        ("ON", PEOPLE[1], STATIONS[1]),
        ("ON", PEOPLE[2], STATIONS[2]),
    ]
    row = [{"reading": reading, "spur": None, "ti": None} for reading in readings]
    return {it["id"]: list(row) for it in items_to_label}
