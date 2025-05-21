categories = {
    "Walk": 0,
    "Graze": 1,
    "Browse": 2,
    "Head Up": 3,
    "Auto-Groom": 4,
    "Trot": 5,
    "Run": 6,
    "Occluded": 7
}

with open("output.pbtxt", "w") as file:
    for name, _id in categories.items():
        file.write(f'item {{\n  name: "{name}"\n  id: {_id}\n}}\n')