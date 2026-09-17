"""Read-only mmap story access; no torch import or full token-array allocation."""
import json
from pathlib import Path
import numpy as np


class StoryDataset:
    def __init__(self, folder, split):
        folder = Path(folder)
        if split not in ('train', 'validation', 'test'):
            raise ValueError('unknown split')
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
        if not manifest.get('complete'):
            raise ValueError('preparation incomplete')
        info = manifest['splits'][split]
        self.tokens = np.memmap(folder / f'{split}.tokens.u16', mode='r', dtype='<u2')
        self.offsets = np.memmap(folder / f'{split}.offsets.u64', mode='r', dtype='<u8')
        self.byte_counts = np.memmap(folder / f'{split}.bytes.u32', mode='r', dtype='<u4')
        if len(self.offsets) != info['stories'] + 1 or int(self.offsets[-1]) != len(self.tokens):
            raise ValueError('offset/token count mismatch')
        if len(self.tokens) != info['tokens'] or len(self.byte_counts) != info['stories']:
            raise ValueError('manifest/file length mismatch')

    def __len__(self):
        return len(self.offsets)-1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start,end = map(int,self.offsets[index:index+2])
        return self.tokens[start:end]  # BOS + payload + EOS; read-only view

    def close(self):
        """Release Windows file handles; all returned views become invalid."""
        for array in (self.tokens, self.offsets, self.byte_counts):
            array._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
