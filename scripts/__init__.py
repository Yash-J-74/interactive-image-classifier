import os
import subprocess

def main():
    # Interactive patch selection script
    print("\nStep 1: Running Interactive Patch Selection...\n")
    patch_selection_script = "interactive_crop.py"
    if os.path.exists(patch_selection_script):
        subprocess.run(["python", patch_selection_script])
    else:
        print(f"Error: {patch_selection_script} not found.")
        return

    # Model Training script
    print("\nStep 2: Training the Model...\n")
    model_training_script = "model_training.py"
    if os.path.exists(model_training_script):
        subprocess.run(["python", model_training_script])
    else:
        print(f"Error: {model_training_script} not found.")
        return

    # Interactive pixel classification script
    print("\nStep 3: Running Interactive Pixel Classification...\n")
    interactive_classifier_script = "inference.py"
    if os.path.exists(interactive_classifier_script):
        subprocess.run(["python", interactive_classifier_script])
    else:
        print(f"Error: {interactive_classifier_script} not found.")
        return

if __name__ == "__main__":
    main()
