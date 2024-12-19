import os
import subprocess

# TODO: Add logging mechanism instead of printing to console

def main():

    print("\nRunning Interactive Patch Selection...\n")
    patch_selection_script = "scripts/interactive_crop.py"
    if os.path.exists(patch_selection_script):
        subprocess.run(["python", patch_selection_script])
    else:
        print(f"Error: {patch_selection_script} not found.")
        return

    print("\Training the Model...\n")
    model_training_script = "scripts/model_training.py"
    if os.path.exists(model_training_script):
        subprocess.run(["python", model_training_script])
    else:
        print(f"Error: {model_training_script} not found.")
        return

    print("\nRunning Interactive Pixel Classification...\n")
    interactive_classifier_script = "scripts/inference.py"
    if os.path.exists(interactive_classifier_script):
        subprocess.run(["python", interactive_classifier_script])
    else:
        print(f"Error: {interactive_classifier_script} not found.")
        return

if __name__ == "__main__":
    main()
