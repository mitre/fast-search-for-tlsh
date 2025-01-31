# fast-tlsh-rs


Note that the library is designed for "old" TLSH digests.


These are the same as "new" TLSH digests, except, unlike "new" TLSH digests, they don't start with the "T1".

If all your TLSH digests start with "T1", then you have the new format. To convert them to the old format, just drop the first two bytes. E.g.

```sh
grep -Evo '^T1.+' file_with_newline_separated_new-format_digests > \
    file_with_newline_separated_old-format_digests
```

