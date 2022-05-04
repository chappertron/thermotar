import unittest

from thermotar.utils import parse_multi_chunk
from thermotar.multichunk import MultiChunk

class TestParseMultiChunk(unittest.TestCase):
    
    def test_parse_lmp_chunks(self):
        parser = parse_multi_chunk.parse_lmp_chunks("./tests/test_files/temp.oxytop")
        # print(parser.data.head())
        # print(parser.data.index)
        self.assertEqual(True,True)

    def test_create_multi_chunk(self):
        chunk = MultiChunk.create_multi_chunks("./tests/test_files/temp.oxytop")
        chunk_no = 32.0
        print(f"chunk={chunk_no}\n",chunk.data[chunk.data.Chunk == chunk_no ])
        chunk.zero_to_nan(val = 0.0, col='Ncount')

        print(chunk.data.head())
        
        flat_chunk = chunk.flatten_chunk()
        print(flat_chunk.data)
if __name__ == '__main__':
    unittest.main()
