import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class InteractiveRegionSelector:
    def __init__(self, image, crop_size=32):
        self.image = image
        self.crop_size = crop_size
        self.foreground_points = []
        self.background_points = []
        self.history = []  # To track all clicks
        self.is_selection_complete = False  # Flag for done action
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.ax.set_title("Left-click for Foreground, Right-click for Background")
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.foreground_patch_folder = "cropped/foreground"
        self.background_patch_folder = "cropped/background"
        os.makedirs(self.foreground_patch_folder, exist_ok=True)
        os.makedirs(self.background_patch_folder, exist_ok=True)

    def on_click(self, event):
        if event.xdata is None or event.ydata is None or self.is_selection_complete:
            return  # Ignores clicks outside the image area or after Done is pressed

        x, y = int(event.xdata), int(event.ydata)

        # Crop patch around the clicked point
        x_start = max(0, x - self.crop_size // 2)
        y_start = max(0, y - self.crop_size // 2)
        x_end = min(self.image.shape[1], x + self.crop_size // 2)
        y_end = min(self.image.shape[0], y + self.crop_size // 2)
        patch = self.image[y_start:y_end, x_start:x_end]

        if patch.shape[0] == self.crop_size and patch.shape[1] == self.crop_size:
            if event.button == 1:  # Left click: Foreground
                self.foreground_points.append((x, y))
                self.history.append(("foreground", (x, y)))
                patch_path = os.path.join(self.foreground_patch_folder, f"patch_{len(self.foreground_points)}.png")
                cv2.imwrite(patch_path, patch)
                print(f"Foreground point saved at ({x}, {y})")

            elif event.button == 3:  # Right click: Background
                self.background_points.append((x, y))
                self.history.append(("background", (x, y)))
                patch_path = os.path.join(self.background_patch_folder, f"patch_{len(self.background_points)}.png")
                cv2.imwrite(patch_path, patch)
                print(f"Background point saved at ({x}, {y})")

        # Update plot
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.ax.set_title("Left-click for Foreground, Right-click for Background")
        for point in self.foreground_points:
            self.ax.plot(point[0], point[1], "go", markersize=5)  # Green point
        for point in self.background_points:
            self.ax.plot(point[0], point[1], "ro", markersize=5)  # Red point
        plt.draw()

    def undo(self):
        if self.history:
            last_action = self.history.pop()  # Remove the last action from history
            point_type, point = last_action

            if point_type == "foreground" and point in self.foreground_points:
                self.foreground_points.remove(point)
                print(f"Removed last foreground point: {point}")
            elif point_type == "background" and point in self.background_points:
                self.background_points.remove(point)
                print(f"Removed last background point: {point}")

            # Update the plot after undoing
            self.update_plot()
        else:
            print("No points to undo.")

    def crop_patches(self):
        print("Interactive region selection complete.")
        return self.foreground_points, self.background_points
    
    def done(self, event=None):
        print("Selection completed. Closing the interface...")
        self.is_selection_complete = True
        plt.close(self.fig)
        
    def save_patches(self, foreground_patches, background_patches, output_dir="cropped"):
        os.makedirs(output_dir, exist_ok=True)
        fg_dir = os.path.join(output_dir, "foreground")
        bg_dir = os.path.join(output_dir, "background")
        os.makedirs(fg_dir, exist_ok=True)
        os.makedirs(bg_dir, exist_ok=True)

        for i, patch in enumerate(foreground_patches):
            if patch.shape[0] == self.crop_size and patch.shape[1] == self.crop_size:
                cv2.imwrite(os.path.join(fg_dir, f"foreground_{i}.png"), patch)

        for i, patch in enumerate(background_patches):
            if patch.shape[0] == self.crop_size and patch.shape[1] == self.crop_size:
                cv2.imwrite(os.path.join(bg_dir, f"background_{i}.png"), patch)

        print(f"Foreground patches saved to {fg_dir}")
        print(f"Background patches saved to {bg_dir}")



# Main Execution
if __name__ == "__main__":

    # TODO: Take image input from UI
    # TODO: Make patch size a variable 
    file_path = input("Enter the image file path: ")
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Could not load image.")
        exit()

    selector = InteractiveRegionSelector(image, crop_size=32)

    # Undo Button
    ax_undo = plt.axes([0.7, 0.01, 0.1, 0.05])  # x, y, width, height
    btn_undo = Button(ax_undo, "Undo")
    btn_undo.on_clicked(lambda event: selector.undo())

    # Done Button
    ax_done = plt.axes([0.81, 0.01, 0.1, 0.05])  # x, y, width, height
    btn_done = Button(ax_done, "Done")
    btn_done.on_clicked(selector.done)

    plt.show()

    # Get the points
    foreground_patches, background_patches = selector.crop_patches()
