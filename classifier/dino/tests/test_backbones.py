"""패치 규격과 register 제외를 다운로드 없이 CPU로 검증한다."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from classifier.dino.model import DamageDINO, backbone_layout, pooled_features


class BackboneTests(unittest.TestCase):
    def test_register_tokens_do_not_enter_patch_pooling(self):
        # register가 매우 커도 패치 평균·최댓값에 섞이면 안 된다.
        hidden = torch.tensor([[[10., 20.], *([[1000., 2000.]] * 4), [1., 4.], [3., 2.]]])
        torch.testing.assert_close(pooled_features(hidden, 4), torch.tensor([[10., 20., 2., 3., 3., 4.]]))

    def test_dinov2_features_remain_identical_without_registers(self):
        hidden = torch.tensor([[[10., 20.], [1., 4.], [3., 2.]]])
        torch.testing.assert_close(pooled_features(hidden), torch.tensor([[10., 20., 2., 3., 3., 4.]]))

    def test_cls_mean_excludes_registers_and_retains_all_patch_gradients(self):
        hidden = torch.tensor([[[10., 20.], *([[1000., 2000.]] * 4), [1., 4.], [3., 2.]]], requires_grad=True)
        actual = pooled_features(hidden, 4, "cls_mean")
        torch.testing.assert_close(actual, torch.tensor([[10., 20., 2., 3.]]))
        actual.sum().backward()
        torch.testing.assert_close(hidden.grad[:, 1:5], torch.zeros(1, 4, 2))
        torch.testing.assert_close(hidden.grad[:, 5:], torch.full((1, 2, 2), .5))
        with self.assertRaises(ValueError):
            pooled_features(hidden, 4, "unknown")

    def test_pooling_controls_head_and_checkpoint_compatibility(self):
        # 다운로드 없이 실제 PEFT 경로와 head strict loading을 확인한다.
        from transformers import DINOv3ViTConfig

        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = DINOv3ViTConfig(hidden_size=768, num_hidden_layers=12,
                                             patch_size=16, num_register_tokens=4)
                self.blocks = torch.nn.ModuleList([
                    torch.nn.ModuleDict({"q_proj": torch.nn.Linear(2, 2), "v_proj": torch.nn.Linear(2, 2)})
                    for _ in range(12)])

            def forward(self, pixel_values):
                return SimpleNamespace(last_hidden_state=torch.ones(len(pixel_values), 6, 768))

        cfg = {"model_type": "dinov3_vit", "model_id": "unused", "revision": "unused",
               "input_size": 16, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .1}
        with patch("classifier.dino.model.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyBackbone()):
            legacy = DamageDINO(cfg)
            legacy_copy = DamageDINO(cfg)
            reduced = DamageDINO({**cfg, "pooling": "cls_mean"})
            reduced_copy = DamageDINO({**cfg, "pooling": "cls_mean"})
        self.assertEqual(legacy.structure["head"], [2304, 2])
        self.assertEqual(reduced.structure["head"], [1536, 2])
        legacy_copy.load_trainable(legacy.trainable_state())
        reduced_copy.load_trainable(reduced.trainable_state())
        pixels = torch.zeros(1, 3, 16, 16)
        torch.testing.assert_close(legacy(pixels), legacy_copy(pixels))
        torch.testing.assert_close(reduced(pixels), reduced_copy(pixels))
        with self.assertRaises(RuntimeError):
            reduced_copy.load_trainable(legacy.trainable_state())

    def test_each_backbone_requires_its_own_patch_multiple(self):
        for family, patch, registers, size in [("dinov2", 14, 0, 1022), ("dinov3_vit", 16, 4, 1024)]:
            actual = SimpleNamespace(model_type=family, patch_size=patch, hidden_size=768,
                                     num_hidden_layers=12, num_register_tokens=registers)
            cfg = {"model_type": family, "input_size": size}
            self.assertEqual(backbone_layout(actual, cfg), (patch, registers))
            with self.assertRaises(ValueError):
                backbone_layout(actual, {**cfg, "input_size": size + 1})
            actual.num_register_tokens += 1
            with self.assertRaises(ValueError):
                backbone_layout(actual, cfg)


if __name__ == "__main__":
    unittest.main()
