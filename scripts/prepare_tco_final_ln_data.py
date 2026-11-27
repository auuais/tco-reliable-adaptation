"""Build separate five-view Stanford RGB/prior and evaluation-only surface packets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
from PIL import Image

from download_stanford12_direction import download
from native_lora_direction_inputs import frame_path, read_window


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_final_global_layernorm_v1"
DEFAULT_DATA = Path("E:/ResearchData/TCO/data/stanford12")
DEFAULT_PACKETS = Path("E:/ResearchData/TCO/preparation") / STUDY
OUTPUT = ROOT / "results" / STUDY
ARCHIVE = Path("E:/ResearchData/TCO/research_results")
DEV = "apt1/kitchen"
APT2 = ("apt2/living", "apt2/luke")
WIDTH = 518
DEPTH_RANGE_M = (0.2, 10.0)
EXPOSURE_SUFFIXES = ("json", "jsonl", "csv", "md", "txt", "log", "yaml", "yml", "tsv")


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for data in iter(lambda: src.read(4 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def array_signature(arrays):
    # Identical key, shape, endianness and raw-byte encoding as run_tco_frozen_prefix.signature.
    h = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        array = np.ascontiguousarray(array)
        h.update(repr((name, array.shape, array.dtype.str)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing receipt differs: {path}")
        return
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(arrays) or any(
                previous[k].dtype != v.dtype or not np.array_equal(previous[k], v) for k, v in arrays.items()
            ):
                raise ValueError(f"Existing packet differs: {path}")
    else:
        temp = path.with_suffix(".npz.partial")
        with temp.open("wb") as out:
            np.savez_compressed(out, **arrays)
        temp.replace(path)
    return sha_file(path)


def calibration(raw):
    entries = dict(line.split(" = ", 1) for line in raw.decode().splitlines() if line.strip())
    size = lambda cam: tuple(int(entries[f"m_{cam}{axis}"]) for axis in ("Width", "Height"))
    matrix = lambda cam, kind: np.fromstring(entries[f"m_calibration{cam}{kind}"], sep=" ").reshape(4, 4)
    if int(entries["m_depthShift"]) != 1000:
        raise ValueError("Unexpected depth scale")
    for name in ("Color", "Depth"):
        if not np.allclose(matrix(name, "Extrinsic"), np.eye(4), rtol=0, atol=1e-8):
            raise ValueError("Published optical frames differ. Verify extrinsic convention before extending.")
    color, depth = size("color"), size("depth")
    raster = (WIDTH, int(round(color[1] * WIDTH / color[0] / 14)) * 14)
    sx, sy = np.array(raster, dtype=np.float64) / color
    mapping = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
    return color, depth, raster, mapping @ matrix("Color", "Intrinsic")[:3,:3], matrix("Depth", "Intrinsic")[:3,:3]


def xyz_from_axial(pixels, depths, intrinsic):
    uv1 = np.column_stack((pixels, np.ones(len(pixels))))
    return (uv1 @ np.linalg.inv(intrinsic).T) * depths[:, None]


def apply_pose(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def dense_reference(depth_mm, kd, kc, raster, c2w, depth_to_color=None):
    """Scatter original surface samples to nearest model pixels, retaining the frontmost."""
    if depth_mm.ndim != 2 or depth_mm.dtype.kind not in "ui" or np.max(depth_mm) > 65535:
        raise ValueError("Expected the original 16-bit axial depth raster")
    z = depth_mm.astype(np.float64) * 0.001
    minimum, maximum = DEPTH_RANGE_M
    yy, xx = np.nonzero((z >= minimum) & (z <= maximum))
    p = xyz_from_axial(np.column_stack((xx, yy)), z[yy, xx], kd)
    if depth_to_color is not None:
        p = apply_pose(p, depth_to_color)
    p = p[(p[:, 2] >= minimum) & (p[:, 2] <= maximum)]
    uvz = p @ kc.T
    xy = uvz[:, :2] / uvz[:, 2:]
    pix = np.rint(xy).astype(np.int64)
    inside = ((pix >= 0) & (pix < raster)).all(axis=1)
    pix, p, xy = pix[inside], p[inside], xy[inside]
    if not len(p):
        raise ValueError("No projected reference points")
    ids = pix[:, 1] * raster[0] + pix[:, 0]
    # Within a pixel, sort by axial distance and then by original sensor scan order.
    order = np.lexsort((np.arange(len(ids)), p[:, 2], ids))
    ids_sorted = ids[order]
    selected = order[np.r_[True, ids_sorted[1:] != ids_sorted[:-1]]]
    w,h = raster
    mask = np.zeros((h,w), dtype=bool)
    surface = np.zeros((h,w,3), dtype=np.float32)
    u,v = pix[selected].T
    mask[v,u] = True
    surface[v,u] = apply_pose(p[selected], c2w).astype(np.float32)
    err = np.max(np.abs(xy[selected] - pix[selected]))
    return surface, mask, float(err)


def fixture():
    k = np.eye(3)
    pose = np.eye(4)
    pose[:3,3] = (.15,-.20,.30)
    depth = np.array([[0,1000,3000],[65535,2500,900]], dtype=np.uint16)
    xyz, mask, error = dense_reference(depth, k, k, (3,2), pose)
    assert mask.tolist() == [[False,True,True],[False,True,True]]
    assert error == 0 and np.allclose(xyz[1,1], [2.65,2.3,2.8])
    assert np.allclose(apply_pose(xyz[mask], np.linalg.inv(pose))[:,2], depth[mask] / 1000.)
    k_small = np.diag([.15,.15,1.])
    selected, one, _ = dense_reference(depth, k, k_small, (3,2), np.eye(4))
    assert int(one.sum()) == 1 and np.allclose(selected[0,0], [1.8,.9,.9])
    rot = np.eye(4)
    rot[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    xyz, _, _ = dense_reference(depth,k,k,(3,2),rot)
    assert np.allclose(xyz[1,1], [-2.5,2.5,2.5])
    move = np.eye(4)
    move[0,3] = 1.
    xyz,mask,_ = dense_reference(np.array([[1000,0,0]],np.uint16),k,k,(3,1),pose,move)
    assert mask.tolist()==[[False,True,False]] and np.allclose(xyz[0,1],[1.15,-.20,1.30])
    return dict(identity=True, inverse_pose=True, translated_depth_camera=True, rotated_world=True,
                mm_to_m=True, collision_nearest=True, invalid_depth=True)


def build_room(archive_path, room, packet_root):
    source_hashes = {}
    with zipfile.ZipFile(archive_path) as z:
        def read(name):
            data = z.read(name)
            source_hashes[name] = hashlib.sha256(data).hexdigest()
            return data

        capture, frames, poses, baseline, stride = read_window(z, room, set(z.namelist()))
        read(room + "/split.txt")
        color, depth, raster, kc, kd = calibration(read(room + "/info.txt"))
        bundle = {key: [] for key in ("images","extrs","intrs","valid_mask","pointclouds")}
        quality = []
        for f in frames.tolist():
            pose = poses[f]
            read(frame_path(room,f,"pose.txt"))
            with Image.open(io.BytesIO(read(frame_path(room,f,"color.jpg")))) as im:
                if im.size != color:
                    raise ValueError("Color image metadata mismatch")
                resize = Image.Resampling.LANCZOS if max(np.divide(raster,color)) < 1 else Image.Resampling.BICUBIC
                rgb = np.asarray(im.convert("RGB").resize(raster,resize)).transpose(2,0,1)
            depth_bytes = read(frame_path(room,f,"depth.png"))
            with Image.open(io.BytesIO(depth_bytes)) as im:
                if im.size != depth or im.mode not in ("I;16","I","I;16L","I;16B"):
                    raise ValueError("Depth image metadata mismatch")
                src = np.asarray(im)
            xyz,mask,err = dense_reference(src,kd,kc,raster,pose)
            w2c = np.linalg.inv(pose)
            # Independent round-trip uses the stored FP32 world surface, inverse pose,
            # and a pinhole division. Pixel difference includes the half-pixel z-buffer association.
            y,x = np.nonzero(mask)
            select = np.arange(0,len(y),max(len(y)//5000,1))
            p = apply_pose(xyz[y[select],x[select]].astype(np.float64), w2c)
            uv = p @ kc.T
            r = float(np.max(np.abs(uv[:,:2]/uv[:,2:] - np.column_stack((x[select],y[select])))))
            if len(y) < 1000 or r > .5001 or err > .50001 or np.ptp(x) < raster[0]*.2 or np.ptp(y)<raster[1]*.2:
                raise ValueError(f"Invalid raster geometry for {room} frame {f}")
            bundle["images"].append(rgb.astype(np.float32)/np.float32(255))
            bundle["extrs"].append(w2c[:3].astype(np.float32))
            bundle["intrs"].append(kc.astype(np.float32))
            bundle["valid_mask"].append(mask)
            bundle["pointclouds"].append(xyz)
            quality.append(dict(frame=f,valid_pixels=int(mask.sum()),fraction=float(mask.mean()),
                                max_correspondence_px=err,fp32_roundtrip_max_px=r))
    model = {k:np.stack(bundle[k]) for k in ("images","extrs","intrs")}
    truth = {k:np.stack(bundle[k]) for k in ("valid_mask","pointclouds")}
    for arr in (*model.values(), *truth.values()):
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite data")
    # The reference-only cross-view check excludes the model entirely.
    overlap=[]
    for i in range(4):
        mask=truth["valid_mask"][i]
        y,x=np.nonzero(mask)
        pts=truth["pointclouds"][i,y[::29],x[::29]]
        target_w2c=model["extrs"][i+1].astype(np.float64)
        points=apply_pose(pts.astype(np.float64),target_w2c)
        positive=points[:,2]>0
        points=points[positive]
        uvz=points @ kc.T
        uv=np.rint(uvz[:,:2]/uvz[:,2:]).astype(np.int64)
        within=((uv>=0)&(uv<raster)).all(axis=1)
        points,uv=points[within],uv[within]
        valid=truth["valid_mask"][i+1,uv[:,1],uv[:,0]]
        uv=uv[valid]
        ref=apply_pose(truth["pointclouds"][i+1,uv[:,1],uv[:,0]].astype(np.float64),target_w2c)
        consistent=np.abs(points[valid,2]-ref[:,2])<=.05
        fraction=float(np.sum(consistent)/len(pts))
        if fraction<.05:
            raise ValueError(f"Adjacent reference views do not overlap in {room}")
        overlap.append(fraction)
    slug=room.replace("/","__")+".npz"
    return dict(scene=room, environmental_site=room.split("/")[0],capture=capture,
                frames=frames.tolist(), stride=stride,baseline_m=float(baseline),width=WIDTH,
                raster=list(raster),source_sha256=source_hashes, color_size=list(color),depth_size=list(depth),
                input_sha256=array_signature(dict(rgb=model["images"],poses=model["extrs"],intrinsics=model["intrs"])),
                reference_sha256=array_signature(dict(mask=truth["valid_mask"],surface=truth["pointclouds"])),
                model_file="model/"+slug,reference_file="reference/"+slug,
                model_npz_sha256=save_npz(packet_root/"model"/slug,model),
                reference_npz_sha256=save_npz(packet_root/"reference"/slug,truth),
                adjacent_reference_overlap_at_50mm=overlap, reference_pixel_checks=quality)


def audit_exposure(rooms, archive_root):
    """Search identities including normalized separators, and distinguish directory inventories from scores."""
    # Restrict candidate names to complete components; negative lookahead is avoided for rg's default engine.
    sep = r"[\\/_ .-]+"
    alternatives = []
    for room in rooms:
        environment, leaf = room.split("/")
        environment_alias = r"(?:apt|apartment)[_ .-]*2" if environment == "apt2" else r"office[_ .-]*1"
        alternatives.append(environment_alias + sep + re.escape(leaf))
        if leaf.startswith("gates") or leaf == "manolis":
            alternatives.append(r"\b" + re.escape(leaf))
    pattern = "(?i)(?:"+"|".join(alternatives)+")"+"(?:[^A-Za-z0-9]|$)"
    matcher = re.compile(pattern)
    scopes = {"manifests":ROOT/"manifests","results":ROOT/"results","reports":ROOT/"reports",
              "docs":ROOT/"docs","runs":ROOT/"runs","archive":archive_root}
    evidence=[]
    unknown=[]
    for scope,folder in scopes.items():
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        ignored = ["-g", "!**/tco_final_global_layernorm_v1/**", "-g", "!tco_final*",
                   "-g", "!TCO_FINAL_GLOBAL_LAYERNORM_HANDOFF.md"]
        listing = subprocess.run(["rg", "--no-ignore", "--files", "--null", *ignored, str(folder)],
                                 capture_output=True)
        if listing.returncode not in (0, 1):
            raise RuntimeError(listing.stderr.decode(errors="replace")[:1000])
        for word in (s for s in listing.stdout.split(b"\0") if s):
            rel = Path(word.decode("utf-8")).relative_to(folder).as_posix()
            if matcher.search(rel):
                unknown.append("artifact-path " + scope + "/" + rel)
        cmd=["rg","--no-ignore","--files-with-matches","--null"]
        for suffix in EXPOSURE_SUFFIXES:
            cmd+=["-g","*."+suffix]
        cmd += [*ignored, "--", pattern, str(folder)]
        proc=subprocess.run(cmd,capture_output=True)
        if proc.returncode not in (0,1):
            raise RuntimeError(proc.stderr.decode(errors="replace")[:1000])
        for word in sorted(p for p in proc.stdout.split(b"\0") if p):
            path=Path(word.decode("utf-8"))
            rel=path.relative_to(folder).as_posix()
            if "".join(path.suffixes).lower().endswith(".json"):
                values=json.loads(path.read_text(encoding="utf-8-sig"))
                leaves=[]
                def walk(obj, keys):
                    if isinstance(obj, dict):
                        for k,v in obj.items():
                            if matcher.search(k): leaves.append((keys+(k,),"MATCHED_KEY"))
                            walk(v, keys+(k,))
                    elif isinstance(obj, list):
                        for v in obj: walk(v,keys)
                    elif isinstance(obj,str) and matcher.search(obj):
                        leaves.append((keys,obj))
                walk(values,())
                if leaves and all(keys[-1:] == ("rooms",) and val in rooms for keys,val in leaves):
                    evidence.append(dict(scope=scope,path=rel,sha256=sha_file(path),
                                         hits=sorted(set(v for _,v in leaves)),classification="publisher_room_inventory"))
                    continue
            unknown.append(scope+"/"+rel)
    if unknown:
        raise ValueError("Prior room mentions need manual exposure review: "+str(unknown))
    return dict(search_roots=list(scopes), searched_suffixes=list(EXPOSURE_SUFFIXES),
                excluded="Current protocol and new study outputs. Includes ignored prior reports and second-disk snapshots. All file paths searched, document contents searched for listed suffixes.",
                room_pattern=pattern,inventory_only_mentions=evidence,unresolved_mentions=[])


def verify_fixture_and_packet(root, row):
    """Read back both sealed archives separately and verify types, hashes and the exact scored support."""
    model_path = root / row["model_file"]
    ref_path = root / row["reference_file"]
    if sha_file(model_path) != row["model_npz_sha256"] or sha_file(ref_path) != row["reference_npz_sha256"]:
        raise ValueError(f"Archive digest mismatch for {row['scene']}")
    width, height = row["raster"]
    with np.load(model_path, allow_pickle=False) as data:
        expected = {"images": (5, 3, height, width), "extrs": (5, 3, 4), "intrs": (5, 3, 3)}
        if set(data.files) != set(expected):
            raise ValueError("Unexpected key in model-only archive")
        inp = {k: data[k] for k in expected}
        for key, shape in expected.items():
            if inp[key].shape != shape or inp[key].dtype != np.float32 or not np.isfinite(inp[key]).all():
                raise ValueError(f"Invalid input {key}")
        if inp["images"].min() < 0 or inp["images"].max() > 1:
            raise ValueError("RGB out of range")
        signature = array_signature(dict(rgb=inp["images"], poses=inp["extrs"], intrinsics=inp["intrs"]))
        if signature != row["input_sha256"]:
            raise ValueError("Input tensor digest changed")
    with np.load(ref_path, allow_pickle=False) as data:
        if set(data.files) != {"valid_mask", "pointclouds"}:
            raise ValueError("Unexpected key in evaluation archive")
        mask, surface = data["valid_mask"], data["pointclouds"]
        if (mask.dtype != np.bool_ or mask.shape != (5, height, width)
                or surface.dtype != np.float32 or surface.shape != (5, height, width, 3)
                or not np.isfinite(surface).all() or np.any(surface[~mask] != 0)):
            raise ValueError("Invalid reference representation")
        if array_signature(dict(mask=mask, surface=surface)) != row["reference_sha256"]:
            raise ValueError("Reference tensor digest changed")
        if np.count_nonzero(mask, axis=(1, 2)).tolist() != [v["valid_pixels"] for v in row["reference_pixel_checks"]]:
            raise ValueError("Pixel count differs from intake receipt")
    return row["scene"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=["selftest", "download", "dev", "roster", "verify"])
    cli.add_argument("--data", type=Path, default=DEFAULT_DATA)
    cli.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    cli.add_argument("--archive", type=Path, default=ARCHIVE)
    args = cli.parse_args()
    if args.phase == "selftest":
        print(json.dumps(fixture(), indent=2))
        return
    if args.phase == "verify":
        fixture()
        dev = json.loads((OUTPUT / "stanford_development_input.json").read_text())["scene"]
        roster = json.loads((OUTPUT / "stanford_inputs.json").read_text())["rooms"]
        for row in [dev, *roster.values()]:
            print("Verified", verify_fixture_and_packet(args.packets, row))
        return
    if args.phase == "download":
        args.data.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=5) as pool:
            download(args.data, "office1", pool)
        return
    checks = fixture()
    if args.phase == "dev":
        receipt = build_room(args.data / "apt1.zip", DEV, args.packets)
        verify_fixture_and_packet(args.packets, receipt)
        write_json(OUTPUT / "stanford_development_input.json", dict(scene=receipt, fixture=checks))
        print(DEV, receipt["frames"], receipt["model_npz_sha256"])
        return
    with zipfile.ZipFile(args.data / "office1.zip") as archive:
        office = sorted(n.rsplit("/", 1)[0] for n in archive.namelist()
                        if n.startswith("office1/") and n.endswith("/split.txt"))
    if not office:
        raise ValueError("Empty Office 1 room inventory")
    rooms = sorted(set(APT2) | set(office))
    exposure = audit_exposure(rooms, args.archive)
    files = {env: sha_file(args.data / (env + ".zip")) for env in ("apt2", "office1")}
    rows = {}
    for room in rooms:
        env = room.split("/")[0]
        rows[room] = build_room(args.data / (env + ".zip"), room, args.packets)
        verify_fixture_and_packet(args.packets, rows[room])
        print(room, rows[room]["frames"], flush=True)
    write_json(OUTPUT / "stanford_inputs.json", dict(
        study=STUDY, rooms=rows, archive_sha256=files, source="https://graphics.stanford.edu/projects/reloc/",
        license="CC BY-NC-SA 4.0", exposure=exposure, fixture=checks,
        fresh_scope="Previous project evaluations; pretraining exposure unknown.",
        sampling="First metadata-eligible publisher capture, start and stride from existing read_window.",
        geometry=("Original registered RGB and depth optical frames, TCO patch-rounded PIL resize, "
                  "axis-specific half-pixel intrinsics. Native axial depth in [0.2,10] m, nearest-depth "
                  "z-buffer with ties-to-even color-pixel assignment and exact winning sensor XYZ in FP32 "
                  "world metres. Invalid surface slots zero, masked out by evaluator.")))


if __name__ == "__main__":
    main()
