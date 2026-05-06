"""Minimal FlatBuffers binary reader, just the helpers we need.

Hand-rolled to avoid the flatbuffers Python package dependency. All offsets
are little-endian, matching the FlatBuffers spec and what the JS bundle's
ByteBuffer implementation does.
"""

from __future__ import annotations

import struct
from typing import Optional


class ByteBuffer:
    """Read-only view of a FlatBuffer-encoded byte buffer."""

    __slots__ = ("data", "position")

    def __init__(self, data: bytes, position: int = 0) -> None:
        self.data = data
        self.position = position

    # Primitive reads
    def read_int8(self, off: int) -> int:
        return struct.unpack_from("<b", self.data, off)[0]

    def read_uint8(self, off: int) -> int:
        return self.data[off]

    def read_int16(self, off: int) -> int:
        return struct.unpack_from("<h", self.data, off)[0]

    def read_uint16(self, off: int) -> int:
        return struct.unpack_from("<H", self.data, off)[0]

    def read_int32(self, off: int) -> int:
        return struct.unpack_from("<i", self.data, off)[0]

    def read_uint32(self, off: int) -> int:
        return struct.unpack_from("<I", self.data, off)[0]

    def read_float32(self, off: int) -> float:
        return struct.unpack_from("<f", self.data, off)[0]

    def read_float64(self, off: int) -> float:
        return struct.unpack_from("<d", self.data, off)[0]

    # FlatBuffer helpers (mirror the JS ByteBuffer methods)
    def __offset(self, table_pos: int, vtable_field_offset: int) -> int:
        """Return the byte offset (relative to table_pos) of a field, or 0
        if the field is missing from this object's vtable."""
        # vtable_offset = first int32 of the table (signed, often negative)
        vtable = table_pos - self.read_int32(table_pos)
        vtable_size = self.read_int16(vtable)
        if vtable_field_offset >= vtable_size:
            return 0
        return self.read_int16(vtable + vtable_field_offset)

    def field_offset(self, table_pos: int, vtable_field_offset: int) -> int:
        return self.__offset(table_pos, vtable_field_offset)

    def vector_data(self, off: int) -> int:
        """Return the absolute offset of the first element of a vector."""
        return off + self.read_int32(off) + 4

    def vector_len(self, off: int) -> int:
        return self.read_int32(off + self.read_int32(off))

    def indirect(self, off: int) -> int:
        return off + self.read_int32(off)

    def string(self, off: int) -> str:
        off = off + self.read_int32(off)
        n = self.read_int32(off)
        return self.data[off + 4 : off + 4 + n].decode("utf-8")

    @classmethod
    def root_table_offset(cls, data: bytes) -> int:
        """For a FlatBuffer file: bytes 0-3 are an int32 offset to the root table."""
        return struct.unpack_from("<i", data, 0)[0]
