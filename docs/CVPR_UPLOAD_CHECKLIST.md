# CVPR 2027 K-FRAG submission checklist

**Assembled from the uploaded repository archive on 2026-11-16.** This is not a certification of reproducibility, anonymity, or submission eligibility.

## Required author review

- [ ] Keep the main scientific paper within eight pages; references and mandatory LLM Usage Statement are excluded from that page limit under CVPR 2027 rules.
- [ ] Include the mandatory LLM Usage Statement after the main text and before the references. Disclose ChatGPT and Codex use accurately.
- [ ] Upload the anonymous manuscript, supplementary PDF (if supplied separately), and anonymous supplementary-code ZIP according to OpenReview's accepted formats and size limits.
- [ ] Do not link the identifying public GitHub repository as submitted supplementary content.
- [ ] Inspect image and checkpoint metadata, source paths, documents and file archives for identifiable usernames, email addresses, names, institution data or hidden secrets. Text scan alone is not a complete anonymity audit.
- [ ] Ensure the supplementary PDF's methodology agrees with final V4.15 code, figure numbers, tables, and actual experimental protocol.
- [ ] Verify that green regional cells are described as **reconstruction support**, never independent region-level authentication.
- [ ] Disclose the public-verifier vs. historical notebook phase-expansion difference.
- [ ] Do not claim the uploaded synthetic tests regenerate the 800 historical DIV2K trial outcomes.
- [ ] Do not claim baseline raw CSVs or original crop coordinates are included: they were not in the uploaded ZIP.
- [ ] If you have the 800-trial baseline CSVs and crop manifest, add only sanitized and authorized versions, then regenerate a clean checksum list.
- [ ] Review the CVPR 2027 author instructions and meet the supplementary deadline (Nov 23, 2026 AOE).

## Current local validation

- Python source AST parsing: PASS.
- Frozen config structure and three checkpoint SHA256 digests: PASS.
- Neural unit/integration scripts: NOT EXECUTED as part of this archive update (the runtime lacked reedsolo).
- Historical 800-trial paper benchmark: NOT EXECUTED.
- Anonymous text filename/content scan for the known author and repository account strings: PASS (this is not a comprehensive metadata check).

## Files

- `KFRAG_V415_GitHub_Update.zip`: public repo replacement snapshot containing the existing public attribution metadata.
- `KFRAG_V415_CVPR2027_Anonymous_Code.zip`: separate code bundle without citation metadata and the existing qualitative image asset.
- This checklist is supplementary preparation guidance, not a scientific result.
