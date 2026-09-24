"""Training saves reproducible inputs; comparison happens in evaluate.py."""

from pathlib import Path

import train as training
from train import train


def test_short_training_run_saves_model_without_unpaired_claim(tmp_path, capsys):
    output_dir = tmp_path / "ppo_seed_11"

    _, rewards, saved_dir = train(
        total_timesteps=8,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=11,
        episode_length=3,
        output_dir=output_dir,
    )

    output = capsys.readouterr().out
    assert Path(saved_dir) == output_dir
    assert (output_dir / "model.zip").exists()
    metadata = (output_dir / "training_metadata.txt").read_text()
    assert "Seed: 11" in metadata
    assert "Total Timesteps: 8" in metadata
    assert "Steps per Update: 8" in metadata
    assert "Improvement:" not in output
    assert rewards


def test_training_reserves_output_before_expensive_learning(tmp_path, monkeypatch):
    output_dir = tmp_path / "reserved"

    class FakePPO:
        def __init__(self, **kwargs):
            self.num_timesteps = 0

        def learn(self, total_timesteps, callback, progress_bar):
            assert output_dir.is_dir()
            self.num_timesteps = total_timesteps
            return self

        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"test model")

    monkeypatch.setattr(training, "PPO", FakePPO)
    training.train(total_timesteps=8, output_dir=output_dir)

    assert (output_dir / "model.zip").exists()
    assert "Actual Timesteps: 8" in (output_dir / "training_metadata.txt").read_text()
