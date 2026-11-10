# ggufranges.c

`tcc -run tools/ggufranges.c MODEL.gguf PREFIX...` prints the byte ranges
of a GGUF file (`offset bytes`, one tensor a line) that hold the tensors
whose names begin with one of the prefixes, and on the standard error
the file's version, its tensor count with the first and last names, and
how many matched with their size.

It reads the header only: the key and value pairs are skipped by type
(strings and arrays of strings by their lengths), `general.alignment` is
taken when present (32 otherwise), and a tensor's length is the distance
to the next tensor's offset, or to the end of the file for the last.

Its use is `libhostram.so`'s `HOSTRAM_TRANSIENT` ([hostram.md](hostram.md)):
the tensors a GPU takes at the load. `scripts/vram35.sh` sorts the lines
and merges neighbours, so that a chunk between two such tensors is not
kept on the host.

Measured on a 35.21 GiB Q8_0 file of 753 tensors: `blk.22.` to `blk.40.`
and `output.weight` are 348 tensors, 16.35 GiB, two ranges once merged
(the output matrix at the head of the file, the blocks at its tail); one
block is 0.84 GiB, three of its tensors 285,212,672 bytes each.
