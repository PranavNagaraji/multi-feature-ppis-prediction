import os
import pickle
import numpy as np
from tqdm import tqdm


def parse_bfactors(pdb_path):
    """
    Parse per-residue B-factors from a PDB file.

    Strategy:
      - Iterate ATOM records, skip altloc fields other than ' ', 'A', '1'.
      - Group atoms by (chain_id, resseq, icode).
      - For each residue prefer the CA B-factor; fall back to mean of all atoms.
      - Return a list of (chain_id, resseq, icode, bfactor) tuples in file order,
        with one entry per unique residue.

    Returns
    -------
    list of float  – raw B-factors in residue order, or None on parse failure.
    """
    residues = {}   # key: (chain, resseq, icode)  value: list of (name, bfac)
    order = []       # residue keys in first-seen order

    try:
        with open(pdb_path, 'r') as fh:
            for line in fh:
                rec = line[:6].strip()
                if rec not in ('ATOM', 'HETATM'):
                    continue
                altloc = line[16]
                if altloc not in (' ', 'A', '1'):
                    continue
                atom_name = line[12:16].strip()
                chain_id  = line[21]
                resseq    = line[22:26].strip()
                icode     = line[26]
                try:
                    bfac = float(line[60:66])
                except (ValueError, IndexError):
                    continue
                key = (chain_id, resseq, icode)
                if key not in residues:
                    residues[key] = []
                    order.append(key)
                residues[key].append((atom_name, bfac))
    except Exception:
        return None

    if not order:
        return None

    bfactors = []
    for key in order:
        atoms = residues[key]
        ca_vals = [b for name, b in atoms if name == 'CA']
        if ca_vals:
            bfactors.append(ca_vals[0])
        else:
            bfactors.append(float(np.mean([b for _, b in atoms])))

    return bfactors


def normalise_bfactors(raw):
    """
    Percentile-robust normalisation to [0, 1].
    Falls back to 0.5 array if the distribution is degenerate.
    """
    arr = np.array(raw, dtype=np.float32)
    p5  = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    span = p95 - p5

    # Degenerate: all B-factors essentially identical
    if span < 1e-3 or (arr.max() - arr.min()) < 1e-3:
        return None  # caller will use fallback

    normed = np.clip((arr - p5) / (span + 1e-6), 0.0, 1.0)
    return normed.astype(np.float32)


def main():
    dataset_path = "./Dataset/"
    output_dir   = "./Feature/bfactor/"
    pdb_dir      = "./PDB/"
    os.makedirs(output_dir, exist_ok=True)

    datasets = ["Train_335.pkl", "Test_60.pkl", "Test_315-28.pkl", "UBtest_31-6.pkl"]
    protein_sequences = {}

    for ds_name in datasets:
        ds_path = os.path.join(dataset_path, ds_name)
        if not os.path.exists(ds_path):
            continue
        with open(ds_path, "rb") as f:
            data = pickle.load(f)
        for pid, val in data.items():
            if pid == '2j3rA':
                continue
            protein_sequences[pid] = val[0]

    print(f"Loaded {len(protein_sequences)} unique protein IDs.")

    success_count = 0
    fallback_ids  = []

    for pid, seq in tqdm(protein_sequences.items(), desc="Generating B-factor features"):
        out_path = os.path.join(output_dir, f"{pid}.npy")
        if os.path.exists(out_path):
            success_count += 1
            continue

        seq_len  = len(seq)
        pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")

        def save_fallback(reason=""):
            arr = np.full((seq_len,), 0.5, dtype=np.float32)
            np.save(out_path, arr)
            fallback_ids.append(pid)
            if reason:
                print(f"  [fallback] {pid}: {reason}")

        if not os.path.exists(pdb_path):
            save_fallback("PDB missing")
            continue

        raw = parse_bfactors(pdb_path)
        if raw is None or len(raw) == 0:
            save_fallback("parse failed or empty")
            continue

        normed = normalise_bfactors(raw)
        if normed is None:
            save_fallback("degenerate B-factor distribution")
            continue

        # Length mismatch: alignment is unreliable (gaps may be interior).
        # Write neutral 0.5 for the whole protein instead of truncate/pad.
        if len(normed) != seq_len:
            print(f"  [len mismatch] {pid}: PDB has {len(normed)} residues, sequence has {seq_len}")
            save_fallback("length mismatch — interior gaps likely, alignment unsafe")
            continue

        np.save(out_path, normed)
        success_count += 1

    print(f"\nSuccessfully generated B-factor features for "
          f"{success_count}/{len(protein_sequences)} proteins.")
    print(f"Fell back to 0.5 for {len(fallback_ids)} proteins:")
    for pid in fallback_ids:
        print(f"  {pid}")


if __name__ == "__main__":
    main()
