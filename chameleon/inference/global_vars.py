_global_dict = {}
    
def set_value(key, value):
    global _global_dict
    _global_dict[key] = value
    print(f"Setting {key}")

def get_value(key):
    global _global_dict
    try:
        return _global_dict[key]
    except:
        return KeyError(f"Key {key} not found in global dictionary")