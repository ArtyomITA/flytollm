import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import prepare_text as p


class TextTests(unittest.TestCase):
    def test_parser_and_normalized_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'tiny.txt'
            path.write_bytes(b'  A\r\nB  <|endoftext|>\n<|endoftext|>unfinished')
            self.assertEqual(list(p.stories(path)), [('A\r\nB',True),('',True),('unfinished',False)])
            self.assertEqual(p.digest('A\r\n B'),p.digest('A B'))
            self.assertNotEqual(p.digest('A B'),p.digest('a b'))

    def test_audit_no_cross_split_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); out=root/'out'
            valid=root/'valid.txt'; train=root/'train.txt'
            valid.write_text('fragment<|endoftext|>held out<|endoftext|>held  out<|endoftext|>tail',encoding='utf8')
            train.write_text('held\nout<|endoftext|>unique<|endoftext|>unique<|endoftext|>',encoding='utf8')
            with patch.object(p,'OUT',out),patch.object(p,'SOURCES',{'train':train,'valid':valid}),patch.object(p,'emit'):
                p.audit()
            data=json.loads((out/'audit.json').read_text())['sources']
            self.assertEqual(data['valid']['kept'],1)
            self.assertEqual(data['valid']['quarantined'],1)
            self.assertEqual(data['valid']['unterminated'],1)
            self.assertEqual(data['train']['kept'],1)
            self.assertEqual(data['train']['overlap_heldout'],1)
            self.assertEqual(data['train']['duplicates_internal'],1)
            self.assertEqual((out/'train.selection.u8').read_bytes(),bytes([0,1,0]))

    def test_mmap_story_boundaries(self):
        from text_dataset import StoryDataset
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            np.asarray([1,300,2,1,400,401,2],dtype='<u2').tofile(root/'train.tokens.u16')
            np.asarray([0,3,7],dtype='<u8').tofile(root/'train.offsets.u64')
            np.asarray([1,2],dtype='<u4').tofile(root/'train.bytes.u32')
            (root/'manifest.json').write_text(json.dumps({'complete':True,'splits':{'train':{'stories':2,'tokens':7}}}))
            with StoryDataset(root,'train') as ds:
                np.testing.assert_array_equal(ds[1],[1,400,401,2])
                self.assertFalse(ds[0].flags.writeable)
                with self.assertRaises(IndexError): ds[2]


if __name__=='__main__': unittest.main(verbosity=2)
