
import os
import codecs

def remove_bom(base_dir):
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                try:
                    with open(path, 'rb') as f:
                        raw = f.read(3)
                    
                    if raw.startswith(codecs.BOM_UTF8):
                        print(f"Removing BOM from {path}")
                        with open(path, 'rb') as f:
                            content = f.read()
                        
                        # strip BOM
                        content = content[3:]
                        
                        with open(path, 'wb') as f:
                            f.write(content)
                        count += 1
                except Exception as e:
                    print(f"Error processing {path}: {e}")
    print(f"Removed BOM from {count} files.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    remove_bom(base_dir)
