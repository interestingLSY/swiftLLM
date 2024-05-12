import torch

class BlockManager:
    def __init__(self, num_blocks: int, max_seqs_in_block_table: int, max_blocks_per_seq: int, block_size: int):
        self.free_blocks_list = list(range(1, num_blocks+1))
        self.free_blocks_list.reverse()
        self.block_table = torch.empty(
            (max_seqs_in_block_table, max_blocks_per_seq),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True
        )
        self.allocated_lens = torch.zeros((max_seqs_in_block_table,), dtype=torch.int32, device="cpu", pin_memory=True)
        self.block_size = block_size
    
    def _allocate_block(self) -> int:
        if not self.free_blocks_list:
            raise RuntimeError("No free blocks available")
        return self.free_blocks_list.pop()
    
    def _allocate_blocks(self, num_blocks: int) -> list[int]:
        if num_blocks > len(self.free_blocks_list):
            raise RuntimeError("Not enough free blocks available")
        if num_blocks == 0:
            return []
        blocks = self.free_blocks_list[-num_blocks:]
        self.free_blocks_list = self.free_blocks_list[:-num_blocks]
        return blocks
    
    def _free_block(self, block_id: int):
        self.free_blocks_list.append(block_id)
    
    def _free_blocks(self, block_ids: list[int]):
        self.free_blocks_list.extend(block_ids)
    
    def allocate_blocks_for_seqs(self, seq_ids: torch.Tensor, target_lens: torch.Tensor):
        block_needed = (target_lens + (self.block_size-1)) // self.block_size - self.allocated_lens[seq_ids]
        blocks = self._allocate_blocks(torch.sum(block_needed))
        blocks = torch.tensor(blocks, dtype=torch.int32, device="cpu")
        for i, seq_id in enumerate(seq_ids):
            cur_num_blocks = block_needed[i].item()
            if cur_num_blocks == 0:
                continue
            seq_block_ids = blocks[-cur_num_blocks:]
            self.block_table[seq_id, self.allocated_lens[seq_id]: self.allocated_lens[seq_id]+cur_num_blocks] = seq_block_ids
            self.allocated_lens[seq_id] += block_needed[i]
            blocks = blocks[:-cur_num_blocks]
        assert len(blocks) == 0
    
    def free_blocks_for_seqs(self, seq_ids: torch.Tensor):
        for seq_id in seq_ids:
            block_ids = self.block_table[seq_id][:self.allocated_lens[seq_id]]
            self._free_blocks(block_ids.tolist())
            self.allocated_lens[seq_id] = 0
            