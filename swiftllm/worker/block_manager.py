import torch

from .kernels.block_mgmt import set_block_table_and_num_seq_alloc_blocks, unset_block_table_and_num_seq_alloc_blocks

class BlockManager:
    """
    BlockManager - Manage the block table and free blocks on GPU

    This manager records the mapping from (sequence ID, block index) to block 
    ID (which we call `block_table`), and provides methods to allocate and free
    blocks.

    All tables (the block table, the `num_seq_allocated_blocks`, and the free block
    list) are all maintained on the GPU, so that we can leverage custom Triton
    kernels for fast operations.
    """

    def __init__(self, num_blocks: int, max_seqs_in_block_table: int, max_blocks_per_seq: int, block_size: int):
        self.num_free_blocks = num_blocks
        self.num_blocks = num_blocks
        self.block_size = block_size

        # seq_id |-> number of blocks allocated for this sequence
        self.num_seq_allocated_blocks = torch.zeros(
            (max_seqs_in_block_table,),
            dtype=torch.int32,
            device="cuda"
        )
        # (seq_id, block_index) |-> block_id
        self.block_table = torch.empty(
            (max_seqs_in_block_table, max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        # block_id |-> whether this block is free or not
        self.is_block_free = torch.ones(
            (num_blocks,),
            dtype=torch.bool,
            device="cuda"
        )
    
    def _allocate_blocks(self, num_blocks: int) -> torch.Tensor:
        """
        Allocate the requested number of blocks, update relevant status, and
        return the block IDs.
        """
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(f"No enough free blocks available ({self.num_blocks} in total, {self.num_free_blocks} free, {num_blocks} requested)")
        selected_blocks = torch.nonzero(self.is_block_free)[:num_blocks].view(-1)
        self.num_free_blocks -= num_blocks
        self.is_block_free[selected_blocks] = False
        return selected_blocks
    
    def _free_blocks(self, block_ids: torch.Tensor):
        """
        Free the specified blocks, and update relevant status.
        """
        self.num_free_blocks += len(block_ids)
        self.is_block_free[block_ids] = True
    
    def allocate_blocks_for_seqs(self, seq_ids: torch.Tensor, target_lens: torch.Tensor):
        target_num_blocks = (target_lens + (self.block_size-1)) // self.block_size
        assert (self.num_seq_allocated_blocks[seq_ids] <= target_num_blocks).all(), \
            f"Logic error: Some sequences have more blocks already allocated than needed. seq_ids: {seq_ids}, target_lens: {target_lens}, target_num_blocks: {target_num_blocks}, self.num_seq_allocated_blocks: {self.num_seq_allocated_blocks}"
        block_needed = target_num_blocks - self.num_seq_allocated_blocks[seq_ids]
        new_blocks = self._allocate_blocks(torch.sum(block_needed))

        set_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, new_blocks, seq_ids, block_needed)
        
    def free_blocks_for_seqs(self, seq_ids: torch.Tensor):
        self.num_free_blocks += torch.sum(self.num_seq_allocated_blocks[seq_ids])
        unset_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)
