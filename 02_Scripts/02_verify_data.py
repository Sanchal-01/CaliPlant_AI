import os
from PIL import Image # Image verify karne ke liye

def verify_and_clean_data(base_path):
    print("🔍 Data ki safai shuru ho rahi hai...\n")
    
    # Sabhi folders (Poppy, Oak, etc.) ko ek ek karke check karenge
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            print(f"📁 Checking {folder}... ({len(files)} files)")
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    # 1. Image ko khol kar check karo
                    img = Image.open(file_path)
                    img.verify() # Ye check karta hai ki file corrupt toh nahi
                except (IOError, SyntaxError) as e:
                    # 2. Agar kharab hai toh delete karo
                    print(f"  ❌ Corrupt file mili aur delete ki gayi: {file}")
                    os.remove(file_path)
            
            # Final count update
            print(f"  ✅ {folder} ab bilkul saaf hai. Total: {len(os.listdir(folder_path))} images.\n")

# --- EXECUTION ---
if __name__ == "__main__":
    train_dir = "03_Data/train"
    
    if os.path.exists(train_dir):
        verify_and_clean_data(train_dir)
    else:
        print("⚠️ Bhai, 03_Data/train folder nahi mila! Path check kar.")