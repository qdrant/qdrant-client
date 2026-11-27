from io import BufferedReader
from typing import Self


class SnapshotUploadReader(BufferedReader):
    """Iterate binary snapshot data in bounded chunks, not newline-delimited records.

    HTTPX 0.20 and 0.21 use file iteration for multipart bodies. Binary backups
    can contain arbitrarily long records, so the default file iterator is unsafe.
    Newer HTTPX versions call read(size) instead, which BufferedReader supports.
    """

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> bytes:
        chunk = self.read(64 * 1024)
        if not chunk:
            raise StopIteration
        return chunk
