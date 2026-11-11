"""Frozen landmark-classifier inference using the training input representation."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from sign_language.gru_model import SignGRU
from sign_language.hands import HandTracker
from sign_language.preprocessing import CONFIG, normalize_sequence
from sign_language.raw_landmarks import frame_observation


class SignRecognizer:
    def __init__(self, checkpoint: Path, models: Path, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        self.config = state['config']
        if self.config != json.loads(self.checkpoint.with_name('config.json').read_text()):
            raise ValueError('Checkpoint and adjacent configuration disagree')
        if self.config['preprocessing'] != CONFIG:
            raise ValueError('Preprocessing configuration differs from checkpoint')
        architecture = self.config.get('architecture', 'gru')
        if architecture == 'stgcn27-v1':
            from sign_language.stgcn_graph import CONFIG as GRAPH
            from sign_language.stgcn_model import SignSTGCN
            if self.config.get('graph') != GRAPH:
                raise ValueError('Graph configuration differs from checkpoint')
            model_class, source_files = SignSTGCN, ('stgcn_model.py', 'stgcn_graph.py')
        elif architecture == 'gru':
            model_class, source_files = SignGRU, ('gru_model.py',)
        elif architecture == 'gru-mean-pool-v1':
            from sign_language.gru_mean_pool_model import SignMeanPoolGRU
            model_class, source_files = SignMeanPoolGRU, ('gru_model.py', 'gru_mean_pool_model.py')
        else:
            raise ValueError('Unknown checkpoint architecture')
        for name in ('preprocessing.py', *source_files):
            digest = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            if digest != self.config['source_sha256'][name]:
                raise ValueError(f'Trained source differs: {name}')
        self.labels = state['labels']
        if self.labels != json.loads(self.checkpoint.with_name('labels.json').read_text()):
            raise ValueError('Checkpoint and adjacent label map disagree')
        self.names = [name for name, _ in sorted(self.labels.items(), key=lambda item: item[1])]
        self.epoch = state['epoch']
        # Same Windows native-GRU workaround as training/evaluation; CUDA remains enabled.
        torch.backends.cudnn.enabled = self.config.get('cudnn_enabled', False)
        self.model = model_class(**self.config['model']).to(self.device).eval()
        self.model.load_state_dict(state['model'], strict=True)
        self.hand_model = Path(models) / 'hand_landmarker.task'
        self.pose_model = Path(models) / 'pose_landmarker_lite.task'
        for path in (self.hand_model, self.pose_model):
            if not path.is_file():
                raise FileNotFoundError(path)

    def predict_observations(self, arrays, frame_shape):
        features, quality = normalize_sequence(arrays, frame_shape)
        if not quality['frames_with_hands'] or not quality['frames_with_body_anchor']:
            raise ValueError('Hands and shoulders must be visible during the sign; please retry')
        # Prepared training/evaluation inputs were stored as float16 then loaded as float32.
        features = features.astype(np.float16).astype(np.float32)
        # ST-GCN temporal convolutions must see the same 128-step grid as training.
        # True lengths still exclude padded time from recurrent state/pooling.
        # A physically shorter tensor changes convolution-boundary activations.
        padded = np.zeros((CONFIG['max_frames'], features.shape[1]), dtype=np.float32)
        padded[:len(features)] = features
        batch = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(features)], dtype=torch.long)
        with torch.inference_mode(), torch.autocast('cuda', enabled=self.device.type == 'cuda'):
            logits = self.model(batch, lengths).float()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite classifier output')
        scores, ids = logits.softmax(-1).topk(min(5, len(self.names)), dim=-1)
        return {'top5': [{'label': self.names[index], 'score': score}
                         for index, score in zip(ids[0].tolist(), scores[0].tolist())],
                'quality': quality, 'input_frames': len(features)}

    def observe_frames(self, frames, timestamps):
        """Decode nothing, save nothing: observe every captured unmirrored frame once."""
        if len(frames) < 8 or len(frames) != len(timestamps):
            raise ValueError('Capture at least eight frames with matching timestamps')
        shape = frames[0].shape
        started = time.perf_counter()
        observations = []
        with ExitStack() as resources:
            hands = resources.enter_context(HandTracker(self.hand_model))
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.pose_model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1, output_segmentation_masks=False,
                min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
                min_tracking_confidence=.5)
            pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            resources.callback(pose.close)
            previous = -1
            for frame, timestamp in zip(frames, timestamps):
                if frame.shape != shape or timestamp <= previous:
                    raise ValueError('Frame dimensions changed or timestamps did not increase')
                previous = timestamp
                hand_result = hands.detect(frame, timestamp)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                observations.append(frame_observation(hand_result, pose.detect_for_video(image, timestamp)))
        arrays = {key: np.stack([item[key] for item in observations]) for key in observations[0]}
        extraction_seconds = time.perf_counter() - started
        return arrays, shape, {'captured_frames': len(frames),
                              'capture_seconds': (timestamps[-1]-timestamps[0])/1000,
                              'landmark_seconds': extraction_seconds}

    def predict_frames(self, frames, timestamps):
        from sign_language.recognition_diagnostics import hand_visibility_summary
        started = time.perf_counter()
        arrays, shape, metadata = self.observe_frames(frames, timestamps)
        result = self.predict_observations(arrays, shape)
        return {**result, **metadata, 'tracking_summary':hand_visibility_summary(arrays['hand_present']),
                'processing_seconds': time.perf_counter()-started}
