import json


def get_config(key):
    """
    Fetches the value of a given configuration key from a JSON file.

    :param key: The key for which the value is to be fetched.
    :return: The value of the specified key from the JSON file, or None if the key does not exist.
    """
    try:
        # Open the JSON file and load the test_data
        with open("config.json", 'r') as file:
            config_data = json.load(file)

        # Return the value associated with the key
        return config_data.get(key)
    except FileNotFoundError:
        # The JSON file was not found
        print("Config file not found.")
        return None
    except json.JSONDecodeError:
        # The JSON file is not properly formatted
        print("Error decoding JSON.")
        return None
