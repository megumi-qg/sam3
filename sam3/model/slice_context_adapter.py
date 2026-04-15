import torch
import torch.nn as nn


class SliceContextAdapter(nn.Module):
    """Convert neighboring slice features into visual prompt tokens."""

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 256,
        pool_size: int = 2,
        max_context_distance: int = 8,
        max_neighbor_frames: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        if max_context_distance < 1:
            raise ValueError(
                f"max_context_distance must be >= 1, got {max_context_distance}"
            )

        self.output_dim = output_dim
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_context_distance = max_context_distance
        self.max_neighbor_frames = max_neighbor_frames
        self.relative_pos_embed = nn.Embedding(
            2 * max_context_distance + 1, output_dim
        )

    def _select_neighbor_indices(self, num_frames: int, center_idx: int) -> list[int]:
        neighbor_indices = [idx for idx in range(num_frames) if idx != center_idx]
        neighbor_indices.sort(key=lambda idx: (abs(idx - center_idx), idx))
        if self.max_neighbor_frames is not None:
            neighbor_indices = neighbor_indices[: self.max_neighbor_frames]
        neighbor_indices.sort()
        return neighbor_indices

    def forward(
        self, frame_features: torch.Tensor, center_idx: int
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Args:
            frame_features: Tensor of shape [T, C, H, W]
            center_idx: Center slice index in the sampled window

        Returns:
            tokens: Tensor of shape [num_tokens, output_dim]
            neighbor_indices: The slice indices used to build the tokens
        """
        if frame_features.dim() != 4:
            raise ValueError(
                f"Expected frame_features with shape [T, C, H, W], got {frame_features.shape}"
            )

        num_frames = frame_features.shape[0]
        if not (0 <= center_idx < num_frames):
            raise IndexError(
                f"center_idx {center_idx} is out of bounds for {num_frames} frames"
            )

        neighbor_indices = self._select_neighbor_indices(num_frames, center_idx)
        if not neighbor_indices:
            return frame_features.new_zeros((0, self.output_dim)), neighbor_indices

        neighbor_feats = frame_features[neighbor_indices]
        pooled_feats = self.pool(neighbor_feats)
        tokens = pooled_feats.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)

        relative_offsets = torch.tensor(
            [idx - center_idx for idx in neighbor_indices],
            device=frame_features.device,
            dtype=torch.long,
        )
        relative_offsets = relative_offsets.clamp(
            min=-self.max_context_distance, max=self.max_context_distance
        )
        relative_offsets = relative_offsets + self.max_context_distance

        rel_pos_embed = self.relative_pos_embed(relative_offsets).unsqueeze(1)
        tokens = self.norm(tokens + rel_pos_embed)
        tokens = self.dropout(tokens)

        return tokens.reshape(-1, self.output_dim), neighbor_indices
