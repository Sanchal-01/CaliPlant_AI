import os
import random
import shutil

def split_data(train_base, val_base, split_ratio=0.2):
    # 1. Validation folder structure banana
    if not os.path.exists(val_base):
        os.makedirs(val_base)
        print(f"📁 Validation folder ban gaya hai.")

    # Har plant folder (category) ko check karo
    for category in os.listdir(train_base):
        train_folder = os.path.join(train_base, category)
        val_folder = os.path.join(val_base, category)

        if os.path.isdir(train_folder):
            # Validation ke andar plant ka sub-folder (jaise Joshua_Tree) banao
            if not os.path.exists(val_folder):
                os.makedirs(val_folder)

            # Sari photos ki list nikalo aur shuffle (mix) karo
            images = os.listdir(train_folder)
            random.shuffle(images)

            # 20% photos calculate karo
            num_val = int(len(images) * split_ratio)
            val_images = images[:num_val]

            print(f"📦 {category}: {len(images)} total mein se {len(val_images)} images Validation mein ja rahi hain...")

            # Photos ko move karna
            for img in val_images:
                src = os.path.join(train_folder, img)
                dst = os.path.join(val_folder, img)
                shutil.move(src, dst)

    print("\n✅ Splitting Done! Ab tera data training ke liye ready hai.")

if __name__ == "__main__":
    # Path dhyan se check kar lena
    split_data("03_Data/train", "03_Data/validation")