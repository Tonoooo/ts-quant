"""
ts_quant.core.memory_manager — Dynamic VRAM Manager
====================================================

Mengelola memori GPU secara dinamis agar library tidak pernah crash
karena kehabisan VRAM, terlepas dari ukuran GPU (8GB, 16GB, 24GB, 80GB).

Fitur utama:
    - Auto-detect VRAM tersedia saat runtime
    - Safety buffer 20% (tidak menggunakan VRAM hingga penuh)
    - Auto-chunking: memecah data menjadi batch yang muat di VRAM
    - OOM retry: jika out-of-memory, otomatis perkecil batch dan coba lagi
    - CPU fallback: jika GPU tidak tersedia, semua perhitungan di CPU
    - Progress bar per batch
"""

import gc
import math
import warnings
from typing import Any, Callable, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm


class VRAMManager:
    """
    Manajer memori GPU dinamis.

    Secara otomatis mendeteksi VRAM yang tersedia, menghitung ukuran
    batch optimal, dan menjalankan fungsi komputasi secara chunked
    agar tidak pernah melebihi kapasitas VRAM.

    Parameters
    ----------
    device : str
        'cuda' atau 'cpu'. Jika 'cuda' tapi GPU tidak tersedia,
        otomatis fallback ke 'cpu'.
    max_vram_gb : float, optional
        Batas VRAM manual (GB). Berguna jika ingin membatasi
        penggunaan VRAM di bawah kapasitas maksimum. None = auto.
    safety_factor : float
        Faktor keamanan (0-1). Default 0.80, artinya hanya gunakan
        80% dari VRAM yang tersedia.
    verbose : bool
        Tampilkan informasi debug.

    Example
    -------
    >>> manager = VRAMManager(device='cuda')
    >>> results = manager.execute_chunked(
    ...     func=my_gpu_function,
    ...     data_list=list_of_tensors,
    ...     estimate_bytes_per_item=1024*1024,  # 1 MB per item
    ... )
    """

    def __init__(
        self,
        device: str = 'cuda',
        max_vram_gb: Optional[float] = None,
        safety_factor: float = 0.80,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.safety_factor = safety_factor

        # ── Resolve device ──
        if device == 'cuda' and not torch.cuda.is_available():
            if self.verbose:
                warnings.warn(
                    "CUDA tidak tersedia. Fallback ke CPU.",
                    RuntimeWarning,
                )
            self.device = torch.device('cpu')
            self.is_gpu = False
        elif device == 'cuda':
            self.device = torch.device('cuda')
            self.is_gpu = True
        else:
            self.device = torch.device('cpu')
            self.is_gpu = False

        # ── VRAM info ──
        if self.is_gpu:
            props = torch.cuda.get_device_properties(0)
            self.gpu_name = props.name
            self.total_vram = props.total_memory
            if max_vram_gb is not None:
                self.usable_vram = int(max_vram_gb * 1024**3)
            else:
                self.usable_vram = self.total_vram
        else:
            self.gpu_name = "CPU (no GPU)"
            self.total_vram = 0
            self.usable_vram = 0

        if self.verbose:
            self._log_device_info()

    # ──────────────────────────────────────────────────────────
    # Public Methods
    # ──────────────────────────────────────────────────────────

    def get_available_bytes(self) -> int:
        """Mengembalikan jumlah byte VRAM yang tersedia saat ini."""
        if not self.is_gpu:
            return 0
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free_in_reserved = reserved - allocated
        total_free = self.usable_vram - allocated
        available = int(total_free * self.safety_factor)
        return max(available, 0)

    def get_available_gb(self) -> float:
        """Mengembalikan jumlah VRAM tersedia dalam GB."""
        return self.get_available_bytes() / (1024**3)

    def estimate_batch_size(
        self,
        n_items: int,
        bytes_per_item: int,
        overhead_factor: float = 2.5,
    ) -> int:
        """
        Menghitung jumlah item optimal per batch.

        Parameters
        ----------
        n_items : int
            Total jumlah item yang akan diproses (misal: jumlah saham).
        bytes_per_item : int
            Estimasi memori per item (bytes).
            Misal: n_timesteps × n_features × 4 (float32)
        overhead_factor : float
            Faktor pengali untuk memperhitungkan tensor sementara
            yang dibuat selama komputasi. Default 2.5x.

        Returns
        -------
        int
            Jumlah item per batch. Minimum 1.
        """
        if not self.is_gpu:
            # CPU mode: proses semua sekaligus
            return n_items

        available = self.get_available_bytes()
        effective_bytes = int(bytes_per_item * overhead_factor)

        if effective_bytes <= 0:
            return n_items

        batch_size = available // effective_bytes
        batch_size = max(1, min(batch_size, n_items))

        if self.verbose:
            n_batches = math.ceil(n_items / batch_size)
            self._log(
                f"Batch estimate: {batch_size} items/batch, "
                f"{n_batches} batches total "
                f"(VRAM avail: {available / 1024**3:.1f} GB, "
                f"per item: {effective_bytes / 1024**2:.1f} MB)"
            )

        return batch_size

    def execute_chunked(
        self,
        func: Callable,
        data_list: List[Any],
        bytes_per_item: int,
        overhead_factor: float = 2.5,
        desc: str = "Processing",
        **kwargs,
    ) -> List[Any]:
        """
        Menjalankan fungsi pada data secara chunked/batch.

        Jika terjadi OOM, otomatis mengurangi batch size 50% dan retry.
        Hasil dari setiap batch dipindahkan ke CPU RAM segera setelah
        selesai untuk membebaskan VRAM.

        Parameters
        ----------
        func : callable
            Fungsi yang menerima list/batch data dan kwargs.
            Signature: func(batch: List, **kwargs) -> List[Tensor]
        data_list : list
            Data yang akan diproses (misal: list of tensors per saham).
        bytes_per_item : int
            Estimasi memori per item.
        overhead_factor : float
            Faktor pengali overhead.
        desc : str
            Deskripsi untuk progress bar.
        **kwargs
            Argument tambahan yang diteruskan ke func.

        Returns
        -------
        list
            Hasil gabungan dari semua batch.
        """
        n_items = len(data_list)
        if n_items == 0:
            return []

        batch_size = self.estimate_batch_size(
            n_items, bytes_per_item, overhead_factor
        )

        results = []
        max_retries = 5
        current_batch_size = batch_size

        i = 0
        retry_count = 0
        pbar = tqdm(total=n_items, desc=desc, disable=not self.verbose)

        while i < n_items:
            end = min(i + current_batch_size, n_items)
            batch = data_list[i:end]

            try:
                batch_results = func(batch, **kwargs)

                # Pindahkan hasil ke CPU segera
                if isinstance(batch_results, torch.Tensor):
                    batch_results = batch_results.cpu()
                elif isinstance(batch_results, list):
                    batch_results = [
                        r.cpu() if isinstance(r, torch.Tensor) else r
                        for r in batch_results
                    ]

                results.append(batch_results)

                # Bersihkan VRAM
                if self.is_gpu:
                    torch.cuda.empty_cache()
                    gc.collect()

                pbar.update(end - i)
                i = end
                retry_count = 0  # reset retry counter

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    # OOM — kurangi batch size dan retry
                    retry_count += 1
                    if retry_count > max_retries:
                        pbar.close()
                        raise RuntimeError(
                            f"OOM setelah {max_retries} retries. "
                            f"Batch size terakhir: {current_batch_size}. "
                            f"VRAM tidak cukup untuk 1 item sekalipun."
                        ) from e

                    old_size = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)

                    if self.verbose:
                        self._log(
                            f"⚠ OOM! Batch {old_size} → {current_batch_size} "
                            f"(retry {retry_count}/{max_retries})"
                        )

                    # Bersihkan memori sebelum retry
                    if self.is_gpu:
                        torch.cuda.empty_cache()
                        gc.collect()
                else:
                    pbar.close()
                    raise

        pbar.close()
        return results

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pindahkan tensor ke device yang dikelola manager."""
        return tensor.to(self.device)

    def clear_cache(self):
        """Bersihkan VRAM cache."""
        if self.is_gpu:
            torch.cuda.empty_cache()
            gc.collect()

    # ──────────────────────────────────────────────────────────
    # Private Methods
    # ──────────────────────────────────────────────────────────

    def _log(self, msg: str):
        """Print log message."""
        print(f"  [VRAMManager] {msg}")

    def _log_device_info(self):
        """Log informasi device saat inisialisasi."""
        if self.is_gpu:
            self._log(f"GPU: {self.gpu_name}")
            self._log(f"Total VRAM: {self.total_vram / 1024**3:.1f} GB")
            self._log(f"Usable VRAM: {self.usable_vram / 1024**3:.1f} GB")
            self._log(f"Safety factor: {self.safety_factor:.0%}")
            available = self.get_available_gb()
            self._log(f"Available now: {available:.1f} GB")
        else:
            self._log("Mode: CPU (no GPU detected)")

    def __repr__(self) -> str:
        if self.is_gpu:
            return (
                f"VRAMManager(gpu='{self.gpu_name}', "
                f"total={self.total_vram / 1024**3:.1f}GB, "
                f"available={self.get_available_gb():.1f}GB)"
            )
        return "VRAMManager(mode=CPU)"
