import pickle
import sys

def load_pickle_file(file_path):
    """
    Load and return the contents of a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        The unpickled data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded pickle file: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to a motion file if no argument provided
        file_path = "data/motions/humanoid/humanoid_walk.pkl"
    
    data = load_pickle_file(file_path)
    
    if data is not None:
        print(f"\nData type: {type(data)}")
        
        # Print some information about the data
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
                else:
                    print(f"  {key}: type = {type(value)}")
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"First element type: {type(data[0])}")
        elif hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
        else:
            print(f"Data: {data}")
