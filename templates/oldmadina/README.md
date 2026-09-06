# Old Madinah Mushaf — template bank

Everything in this directory is specific to *this* Mushaf: one hand, one
scan set, one page layout. A second Mushaf gets a sibling directory rather
than edits to these files.

| file | passed as | what it is |
| --- | --- | --- |
| `skeleton_templates.json` | `--skeleton-templates` | the letter-shape templates the PAW segmenter aligns against |
| `mark-templates/` | `--mark-template-root` | reviewed component masks, one directory per mark class |
| `segmentation_exceptions.json` | `--segmentation-exceptions` | per-page overrides for words this Mushaf's layout gets wrong |

## Reviewed mark templates

`mark-templates/` is the canonical bank, passed with:

```sh
--mark-template-root templates/oldmadina/mark-templates
```

Each class has its own directory and contains normalized component masks. Waqf
classes use the same layout as ordinary marks. For `waqf_qaf` and
`waqf_jeem`, the files are body templates; the recognizer finds the detached
companion component from its geometry. The specialized recognizers also read
`dot_above_three/` and `alef_hamza_above_attached/` from this bank.

Kasratan has two template roles: `kasratan/*.png` contains the ordinary,
nearly vertically aligned bars, while `kasratan/open/*.png` contains the open
form whose lower bar is shifted to the left. Both forms are exported as the
semantic `kasratan` mark; the role only controls recognition and line-boundary
repair.
