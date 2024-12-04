import json


def print_structure(data, indent=0):
    """
    Recursively prints the structure of the JSON data.
    """
    indent_str = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent_str}{key}: {type(value).__name__}")
            print_structure(value, indent + 2)
    elif isinstance(data, list):
        print(f"{indent_str}List of length {len(data)}")
        if len(data) > 0:
            print_structure(data[0], indent + 2)
    else:
        print(f"{indent_str}{type(data).__name__}")


def main():
    # Load the JSON file
    with open("DAVE_3_shot_val.json", "r") as file:
        data = json.load(file)

    # Print the structure of the JSON data
    print_structure(data)


if __name__ == "__main__":
    main()
