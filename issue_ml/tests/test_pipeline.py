import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from issue_ml.classification.preprocessing import clean, PREPROCESSING_VERSION
from issue_ml.classification.data import prepare
from issue_ml.classification.splitting import temporal_split

NEG, POS = 'bug', 'enhancement'

def row(n, title, body, labels):
    return {'number': n, 'created_at': f'2026-01-{n:02d}T00:00:00Z',
            'title': title, 'body': body, 'labels': labels}


class DataContractTests(unittest.TestCase):

    def test_no_eligible_records_rejected(self):
        with self.assertRaises(ValueError):
            prepare([row(1, 'both', 'body', ['bug', 'enhancement'])], NEG, POS)

    def test_same_labels_rejected(self):
        with self.assertRaises(ValueError):
            prepare([row(1, 'x', 'body', ['bug'])], 'bug', 'bug')

    def test_missing_field_raises_key_error(self):
        bad = {'number': 1, 'created_at': '2026-01-01T00:00:00Z', 'title': 't', 'body': 'b'}
        # 'labels' 字段缺失
        with self.assertRaises(KeyError):
            prepare([bad], NEG, POS)

    def test_content_retained_template_removed(self):
        text = clean('[Bug] Windows crash', '### What issue are you seeing?\nWindows error 42')
        self.assertIn('Windows error 42', text)
        self.assertNotIn('What issue', text)
        self.assertNotIn('[Bug]', text)

    def test_none_title_body_handled(self):
        """clean() 应对 None 值容错，不抛出 AttributeError。"""
        result = clean(None, None)
        self.assertIsInstance(result, str)

    def test_exclusive_labels_and_conflicts(self):
        rows = [row(1, 'same', 'body', ['bug']),
                row(2, 'same', 'body', ['enhancement']),
                row(3, 'both', 'body', ['bug', 'enhancement']),
                row(4, 'valid', 'body', ['bug'])]
        frame, audit = prepare(rows, NEG, POS)
        self.assertEqual(frame.number.tolist(), [4])
        self.assertEqual(audit['conflicting_text_rows_removed'], 2)

    def test_future_title_overlap_removed(self):
        rows = [row(i, 'repeat' if i in (1, 5) else f'title{i}', f'body{i}',
                    ['bug'] if i % 2 else ['enhancement'])
                for i in range(1, 7)]
        frame, _ = prepare(rows, NEG, POS)
        train, test, _ = temporal_split(frame, 0.5)
        self.assertNotIn(5, test.number.tolist())
        self.assertLess(train.created_at.max(), test.created_at.min())

    def test_empty_split_rejected(self):
        frame, _ = prepare([row(1, 'a', 'body', ['bug'])], NEG, POS)
        with self.assertRaises(ValueError):
            temporal_split(frame)

    def test_preprocessing_version_constant_exists(self):
        """PREPROCESSING_VERSION 常量必须存在，training.py 和 prediction.py 均依赖它。"""
        self.assertIsInstance(PREPROCESSING_VERSION, str)
        self.assertTrue(PREPROCESSING_VERSION)


if __name__ == '__main__':
    unittest.main()
