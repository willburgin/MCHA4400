import os, sys, json, numpy as np, cv2
from PIL import Image

TRAIN_PATH = "../data/train"  # adjust if needed

def get_component_dominant_hues(annotated_image_rgb: np.ndarray):
    # Foreground mask (bright/colored)
    bright_mask = annotated_image_rgb.max(axis=2) > 128
    hsv = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]

    # Connected components over foreground (8-neighbourhood)
    num_labels, labels = cv2.connectedComponents(bright_mask.astype(np.uint8), connectivity=8)

    # For each component, pick its dominant (most frequent) hue
    comp_to_hue = {}
    for cid in range(1, num_labels):
        mask = labels == cid
        if not np.any(mask):
            continue
        # dominant hue in this component
        hs = H[mask]
        # bincount requires non-negative ints; H is already 0..179 in OpenCV
        counts = np.bincount(hs)
        dominant_h = int(np.argmax(counts))
        comp_to_hue[cid] = dominant_h

    return comp_to_hue, num_labels - 1

def verify_image(path):
    img = np.array(Image.open(path).convert("RGB"))
    comp_to_hue, n_comp = get_component_dominant_hues(img)

    # Rule: each hue should be used by at most one component (global uniqueness)
    hue_to_components = {}
    for cid, h in comp_to_hue.items():
        hue_to_components.setdefault(h, []).append(cid)

    duplicated_hues = {h: cids for h, cids in hue_to_components.items() if len(cids) > 1}

    return {
        "path": path,
        "num_components": n_comp,
        "assigned_hues": sorted(list(set(comp_to_hue.values()))),
        "component_hues": comp_to_hue,        # component_id -> chosen dominant hue
        "duplicated_hues": duplicated_hues,   # hue value reused across multiple components
        "ok": (len(duplicated_hues) == 0)
    }

def main():
    annotated = [f for f in os.listdir(TRAIN_PATH) if f.endswith("_annotated.png")]
    results = []
    any_fail = False
    for f in sorted(annotated):
        report = verify_image(os.path.join(TRAIN_PATH, f))
        results.append(report)
        if not report["ok"]:
            any_fail = True

    # Pretty summary
    for r in results:
        print(f"\n{r['path']}")
        print(f"  components: {r['num_components']}, assigned H: {r['assigned_hues']}")
        if r["duplicated_hues"]:
            print("  ERROR: hues reused across components:", r["duplicated_hues"])

    # Save a JSON manifest for auditing
    with open("hue_audit.json", "w") as f:
        json.dump(results, f, indent=2)

    if any_fail:
        print("\nFAILED: Hue uniqueness issues detected. See hue_audit.json")
        sys.exit(1)

    print("\nPASS: All annotated images use unique hues per instance.")

if __name__ == "__main__":
    main()
