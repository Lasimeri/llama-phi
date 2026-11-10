# dbg-cmp.sh: where two forward passes part

`dbg-cmp.sh A B [TOL]` reads two llama-debug logs ([`dbg-run.md`](dbg-run.md)),
pairs their tensors in graph order and prints the first eight whose sums
differ by more than TOL relative to max(|reference|, 1) (default 0.01)
or are NaN or infinite, as `tensor I NAME (OP): reference X, card Y`,
then how many tensors were compared. A different tensor name at the same
position ends the comparison ("graph differs"). A is the reference (the
twin's `--native` log), B the run under test.
