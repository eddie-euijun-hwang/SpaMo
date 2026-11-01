from pathlib import Path
import h5py
import numpy as np
import os


def generate_mock_data(output_dir, num_samples=100):
    os.makedirs(output_dir, exist_ok=True)

    # Paths
    vis_path = os.path.join(output_dir, "mock_visual.h5")
    mot_path = os.path.join(output_dir, "mock_motion.h5")
    txt_path = os.path.join(output_dir, "mock_text.h5")

    # Dimensions
    vision_dim = 2048  # Zgodne z FusionModel vision_input_dim lub T5 input
    motion_dim = 1024  # Zgodne z FusionModel motion_input_dim

    print(f"Generowanie {num_samples} próbek w {output_dir}...")

    with (
        h5py.File(vis_path, "w") as f_vis,
        h5py.File(mot_path, "w") as f_mot,
        h5py.File(txt_path, "w") as f_txt,
    ):
        for i in range(num_samples):
            key = f"video_{i:04d}"

            # Random number of frames (e.g., 10 to 50)
            n_frames = np.random.randint(10, 51)

            # Random features
            vis_feat = np.random.randn(n_frames, vision_dim).astype(np.float32)
            mot_feat = np.random.randn(n_frames, motion_dim).astype(np.float32)

            # Random text
            text = f"To jest przykładowe zdanie numer {i}."

            # Save to files
            f_vis.create_dataset(key, data=vis_feat)
            f_mot.create_dataset(key, data=mot_feat)
            f_txt.create_dataset(key, data=text)

    print("Zakończono generowanie danych.")
    print(f"Visual: {vis_path}")
    print(f"Motion: {mot_path}")
    print(f"Text:   {txt_path}")


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent
    generate_mock_data(path / "tests" / "mock_data")
