import argparse
import itertools
import json
import logging
import math
import multiprocessing
import os
import pathlib
import pprint
import random
from datetime import datetime
from itertools import repeat
from typing import Dict, List, Set, Tuple

import dateutil.tz
import geopandas as gpd
import rasterio as rio
from rasterio import mask
from sssl import utils
from tqdm import tqdm

logger = logging.getLogger("__main__")


def find_admin_zone_and_bbox(
    p: str, shp: gpd.GeoDataFrame
) -> Tuple[str, rio.coords.BoundingBox]:
    """
        Bbox in lat,lon degrees since files in EPSG:4326
        https://rasterio.readthedocs.io/en/latest/quickstart.html
        https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html
    :param p:
    :param shp:
    :return:
    """
    ds = rio.open(p)
    return find_admin_zone_ds(ds, shp), ds.bounds


def find_admin_zone_ds(ds: rio.DatasetReader, shp: gpd.GeoDataFrame) -> str:
    """
    :param ds:
    :param shp:     Make sure shapefile is in same CRS as ds!
    :return:
    """
    shp = shp.copy(deep=True)
    shp.geometry = shp.geometry.to_crs(ds.crs)
    overlapping_shp = shp.cx[
        ds.bounds.left : ds.bounds.right, ds.bounds.bottom : ds.bounds.top
    ]
    if len(overlapping_shp) == 0:
        return "NONE"
    elif len(overlapping_shp) == 1:
        return overlapping_shp.ADMIN2.iloc[0]
    else:
        overlaps = []
        for i in range(len(overlapping_shp)):
            masked, _ = rio.mask.mask(
                ds, [overlapping_shp.iloc[i].geometry], filled=False, invert=True
            )
            overlaps.append((overlapping_shp.iloc[i].ADMIN2, masked.mask.sum()))
        max_overlap = max(overlaps, key=lambda t: t[1])
        return max_overlap[0]


def worker_fn(p: str, shp: gpd.GeoDataFrame) -> Tuple[str, str, rio.coords.BoundingBox]:
    z, b = find_admin_zone_and_bbox(p, shp)
    return p, z, b


def compute_admin_zones_coords(
    path_to_tiles: str,
    path_to_shp: str,
    out_dir: str,
    num_workers: int,
    debug: bool = False,
    remove: bool = True,
):
    logger.info("Scanning dir %s" % path_to_tiles)
    fns = []
    for i, entry in tqdm(enumerate(os.scandir(path_to_tiles))):
        if debug and i >= 10000:
            break
        fns.append(entry.path)
    logger.info("Found %s tiles" % len(fns))

    logger.info("Reading first tile from %s" % fns[0])
    first_file = rio.open(fns[0])
    logger.info("Reading shapefile from %s" % path_to_shp)
    shapefile = gpd.read_file(path_to_shp)
    logger.info(
        "Converting shapefile CRS from %s to first tile CRS %s"
        % (shapefile.crs, first_file.crs)
    )
    shapefile.geometry = shapefile.geometry.to_crs(first_file.crs)

    total, chunksize = len(fns), 2000
    logger.info(
        "Finding admin zones and bboxes for tiles with %s workers..." % num_workers
    )
    with multiprocessing.Pool(num_workers) as pool:
        result = pool.starmap(
            worker_fn,
            tqdm(zip(fns, repeat(shapefile)), total=total),
            chunksize=chunksize,
        )

    path2zb, zone2box2p, box2pz, none_tiles = {}, {}, {}, []
    for i, (p, z, b) in tqdm(enumerate(result)):
        filename = pathlib.Path(p).name
        if z != "NONE":
            path2zb[filename] = (z, b)
            zone2box2p.setdefault(z, {}).setdefault(utils.box_to_str(b), []).append(
                filename
            )
            box2pz.setdefault(utils.box_to_str(b), []).append((filename, z))
        else:
            none_tiles.append(filename)
            if remove and not debug:
                os.remove(filename)

    counts = {z: len(zone2box2p[z]) for z in zone2box2p}
    counts["NONE"] = len(none_tiles)
    logger.info("Admin zone distribution:")
    logger.info(pprint.pformat(counts, indent=2))

    logger.info(
        "Writing region counts json to %s" % os.path.join(out_dir, "region_counts.json")
    )
    with open(os.path.join(out_dir, "region_counts.json"), "w") as f:
        json.dump(counts, f)
    logger.info(
        "Writing index json to %s"
        % os.path.join(out_dir, "tilepath_2_adminzone_box.json")
    )
    with open(os.path.join(out_dir, "tilepath_2_adminzone_box.json"), "w") as f:
        json.dump(path2zb, f)
    logger.info(
        "Writing index json to %s"
        % os.path.join(out_dir, "adminzone_2_box_2_tilepath.json")
    )
    with open(os.path.join(out_dir, "adminzone_2_box_2_tilepath.json"), "w") as f:
        json.dump(zone2box2p, f)
    logger.info(
        "Writing index json to %s"
        % os.path.join(out_dir, "box_2_tilepath_adminzone.json")
    )
    with open(os.path.join(out_dir, "box_2_tilepath_adminzone.json"), "w") as f:
        json.dump(box2pz, f)
    logger.info(
        "Writing paths of tiles out-of-region to %s"
        % os.path.join(out_dir, "out_of_region_tiles.json")
    )
    with open(os.path.join(out_dir, "out_of_region_tiles.json"), "w") as f:
        json.dump(none_tiles, f)

    return path2zb, zone2box2p, box2pz, none_tiles


def build_dicts(path2zb, zone2box2p, box2pz, none_tiles, out_dir):

    logger.info("Building latitude-longitude dicts from all boxes")
    all_boxes = list(box2pz.keys())
    all_lats, all_lons = set(), set()
    for box in tqdm(all_boxes):
        all_lats.update(utils.boxstr_to_lats(box))
        all_lons.update(utils.boxstr_to_lons(box))

    all_lats = sorted(list(all_lats), key=lambda x: float(x), reverse=True)
    all_lons = sorted(list(all_lons), key=lambda x: float(x), reverse=False)
    lat_dict = {lat: i for (i, lat) in enumerate(all_lats)}
    lon_dict = {lon: i for (i, lon) in enumerate(all_lons)}

    logger.info("Building date dict from all paths")

    all_end_dates = set()
    for p in tqdm(path2zb.keys()):
        all_end_dates.update([utils.path_to_end_date(p)])
    all_end_dates = sorted(list(all_end_dates))
    all_end_dates = [d.strftime(utils.DATE_FORMAT) for d in all_end_dates]
    date_dict = {d: i for (i, d) in enumerate(all_end_dates)}

    all_admins = sorted(list(zone2box2p.keys()))
    admin_dict = {a: i for (i, a) in enumerate(all_admins)}

    # confirm shaped like somalia
    # lat_lon_date_lens = np.array([
    #     [
    #         len([p for p in lat_lon_date_path[i][j] if p != None])
    #         for j in range(len(all_lons))
    #     ] for i in range(len(all_lats))
    # ])
    # lat_lon_date_lens = np.array(
    #     [[sum([len(lat_lon_date_path[i][j][k]) for k in range(len(all_end_dates))]) for j in range(len(all_lons))]
    #      for i in range(len(all_lats))])
    # plt.close()
    # plt.imshow(lat_lon_date_lens, cmap='hot', interpolation='nearest')
    # plt.show()

    lat_lon_date_path, admin_date_paths = build_maps(
        box2pz.keys(),
        all_admins,
        box2pz,
        zone2box2p,
        {
            "lat_dict": lat_dict,
            "lon_dict": lon_dict,
            "date_dict": date_dict,
            "admin_dict": admin_dict,
        },
    )

    dicts = {
        "all_lats": all_lats,
        "all_lons": all_lons,
        "lat_dict": lat_dict,
        "lon_dict": lon_dict,
        "all_end_dates": all_end_dates,
        "date_dict": date_dict,
        "all_admins": all_admins,
        "admin_dict": admin_dict,
    }
    maps = {
        "lat_lon_date_path": lat_lon_date_path,
        "admin_date_paths": admin_date_paths,
    }

    out_path = os.path.join(out_dir, "dicts.json")
    logger.info("Writing dictionaries to %s" % out_path)
    with open(out_path, "w") as f:
        json.dump(dicts, f)
    out_path = os.path.join(out_dir, "maps.json")
    logger.info("Writing maps to %s" % out_path)
    with open(out_path, "w") as f:
        json.dump(maps, f)

    return dicts, maps


def build_maps(boxes, zones, box2pz, zone2box2p, dicts):
    lat_dict, lon_dict, date_dict, admin_dict = (
        dicts["lat_dict"],
        dicts["lon_dict"],
        dicts["date_dict"],
        dicts["admin_dict"],
    )
    lat_lon_date_path = [
        [[] for _ in range(len(lon_dict))] for _ in range(len(lat_dict))
    ]
    admin_date_paths = [
        [[] for _ in range(len(date_dict))] for _ in range(len(zone2box2p))
    ]
    for boxstr in tqdm(boxes):
        pzlist = box2pz[boxstr]
        tl_lat, tl_lon = utils.boxstr_to_tl_lat_lon(boxstr)
        lati, loni = lat_dict[tl_lat], lon_dict[tl_lon]
        for path, zone in pzlist:
            end_date = utils.path_to_end_date(path, ftstr=True)
            dati = date_dict[end_date]

            datelist = lat_lon_date_path[lati][loni]
            if len(datelist) == 0:
                lat_lon_date_path[lati][loni] = [None for _ in range(len(date_dict))]
            lat_lon_date_path[lati][loni][dati] = path

            if zone in zones:
                admin_date_paths[admin_dict[zone]][dati].append(path)

    return lat_lon_date_path, admin_date_paths


def get_space_positives(
    anchor: str,
    all_boxes: List[str],
    lat_lon_date_path: List,
    dicts: Dict,
    path2zb: Dict,
    distances: Tuple[float, float] = (0.15, 0.4),
    per_distance: Tuple[int, int] = (7, 48),
) -> Set[str]:
    """
    (7, 49): 7 so we have K=8 incl anchor positives for smallest threshold
             48 to have an equally densely represented area for the larger threshold
                (0.8 * 0.8) / (0.3 * 0.3) = 7
                8 / 0.3^2 = M / 0.8^2 => M = 8 * 0.8^2 / 0.3^2 = 8 * 7
                But 8 of smaller area contained in larger area so 8 * 7 - 8 = 48
    """
    # These are indices!
    lats, lons = utils.boxstr_to_tl_lat_lon(anchor)
    anchor_lati, anchor_loni = dicts["lat_dict"][lats], dicts["lon_dict"][lons]
    anchor_lat, anchor_lon = float(dicts["all_lats"][anchor_lati]), float(
        dicts["all_lons"][anchor_loni]
    )
    # anchor_lat, anchor_lon = float(lats), float(lons)
    indices = []
    offset = 0
    space_positive = []
    for (dist, num) in zip(distances, per_distance):

        candidate_positive = []

        incr_offset = True
        while incr_offset:
            offset += 1
            indices = get_offset_indices(
                anchor_lati,
                anchor_loni,
                offset,
                len(dicts["all_lats"]),
                len(dicts["all_lons"]),
            )
            f_indices = filter_out_of_bounds(
                indices, anchor_lat, anchor_lon, dicts, dist
            )

            if len(f_indices) < 1 and offset > 1:
                incr_offset = False
                offset -= 1  # otherwise next dist iteration will skip one offset

            for (lat, lon) in f_indices:

                for (date, p) in enumerate(lat_lon_date_path[lat][lon]):
                    if p is not None:
                        b = utils.box_to_str(path2zb[p][1])
                        if b in candidate_positive or b not in all_boxes or b == anchor:
                            break
                        else:
                            candidate_positive.append(b)

        if len(candidate_positive) < num:
            raise ValueError("Not enough neighbors found!")

        space_positive += candidate_positive

    space_positive = set(space_positive)
    return space_positive


def filter_out_of_bounds(
    indices: List[Tuple[int, int]], alat: float, alon: float, dicts: Dict, limit: float
) -> List[Tuple[int, int]]:
    result = []
    for (lati, loni) in indices:
        lat = float(dicts["all_lats"][lati])
        lon = float(dicts["all_lons"][loni])
        if not sq_out_of_bounds(alat, alon, lat, lon, limit):
            result.append((lati, loni))
    return result


def sq_out_of_bounds(
    anchor_lat: float, anchor_lon: float, lat: float, lon: float, limit: float
) -> bool:
    return math.fabs(lat - anchor_lat) > limit or math.fabs(lon - anchor_lon) > limit


def get_offset_indices(lati, loni, offset, nb_lats, nb_lons):
    indices = []
    for offset_sign in [1, -1]:
        # lat constant, corners inclusive
        lat = lati + offset_sign * offset
        indices += [(lat, lon) for lon in range(loni - offset, loni + offset + 1)]
        # lon constant, corners exclusive
        lon = loni + offset_sign * offset
        indices += [(lat, lon) for lat in range(lati - offset + 1, lati + offset)]
        indices = [
            (lat, lon)
            for (lat, lon) in indices
            if 0 <= lat < nb_lats and 0 <= lon < nb_lons
        ]
    return list(set(indices))


def split_boxes(
    all_boxes: List[str],
    lldp: List,
    dicts: Dict,
    path2zb: Dict,
    args: argparse.Namespace,
) -> Tuple[Set[str], Set[str], Set[str]]:

    shuffled_boxes = random.sample(all_boxes, k=len(all_boxes))
    nb_val, nb_test = int(args.val_fraction * len(all_boxes)), int(
        args.test_fraction * len(all_boxes)
    )

    logger.info("Taking val boxes...")
    anchor_i, not_val_boxes, val_boxes = sample_inf_set(
        args, dicts, lldp, nb_val, path2zb, shuffled_boxes
    )

    logger.info(
        "Wanted %s boxes for val (%s), took %s (%s), with %s anchors sampled"
        % (
            nb_val,
            args.val_fraction,
            len(val_boxes),
            len(val_boxes) / len(all_boxes),
            anchor_i + 1,
        )
    )

    logger.info("Taking test boxes...")
    t_anchor_i, train_boxes, test_boxes = sample_inf_set(
        args, dicts, lldp, nb_test, path2zb, list(not_val_boxes)
    )
    logger.info(
        "Wanted %s boxes for test (%s), took %s (%s)"
        % (
            nb_test,
            args.test_fraction,
            len(test_boxes),
            len(test_boxes) / len(all_boxes),
        )
    )

    assert (
        len(val_boxes.intersection(test_boxes))
        == len(val_boxes.intersection(train_boxes))
        == len(test_boxes.intersection(train_boxes))
        == 0
    )

    utils.plot_tile_loc(
        [utils.boxstr_to_tl_lat_lon(b) for b in val_boxes],
        path_to_shp=args.admin_shapes,
    )
    return val_boxes, test_boxes, train_boxes


def sample_inf_set(
    args, dicts, lldp, nb_val, path2zb, shuffled_boxes
) -> Tuple[int, Set[str], Set[str]]:
    val_boxes = set()
    anchor_i = 0
    not_val_boxes = set(shuffled_boxes)
    for i in itertools.count():
        if len(val_boxes) >= nb_val:
            break
        anchor = next(iter(not_val_boxes))
        anchor_i = i
        new_val_boxes = get_space_positives(
            anchor,
            not_val_boxes,
            lldp,
            dicts,
            path2zb,
            args.distances,
            args.per_distance,
        )
        val_boxes.update(new_val_boxes)
        val_boxes.add(anchor)
        not_val_boxes = not_val_boxes.difference(val_boxes)
    return anchor_i, set(not_val_boxes), val_boxes


def train_val_test_splits(
    zone2box2p,
    box2pz,
    path2zb,
    lat_lon_date_path,
    dicts,
    out_dir,
    args,
):
    # split into in-domain and out-of-domain
    shuffled_regions = random.sample(list(zone2box2p.keys()), k=len(zone2box2p))
    ood_regions, id_regions = (
        shuffled_regions[: args.ood_regions],
        shuffled_regions[args.ood_regions :],
    )
    ood_boxes = list(set([b for zone in ood_regions for b in zone2box2p[zone].keys()]))
    ood_paths = [p for b in ood_boxes for (p, z) in box2pz[b]]
    logger.info(
        "Picked %s of %s regions: %s as OOD test set, with %s of %s tiles (%s/1.0)"
        % (
            args.ood_regions,
            len(zone2box2p),
            ood_regions,
            len(ood_boxes),
            len(box2pz),
            len(ood_boxes) / len(box2pz),
        )
    )

    downstr_val_regions = id_regions[: args.val_downstr]
    downstr_test_regions = id_regions[
        args.val_downstr : args.val_downstr + args.test_downstr
    ]
    downstr_train_regions = id_regions[args.val_downstr + args.test_downstr :]
    logger.info(
        "Picked %s of %s regions: %s as downstream test set"
        % (args.test_downstr, len(zone2box2p), downstr_test_regions)
    )
    logger.info(
        "Picked %s of %s regions: %s as downstream val set"
        % (args.val_downstr, len(zone2box2p), downstr_val_regions)
    )
    logger.info(
        "Picked %s of %s regions: %s as downstream train set"
        % (
            len(zone2box2p) - args.val_downstr - args.test_downstr,
            len(zone2box2p),
            downstr_train_regions,
        )
    )

    id_boxes = list(set([b for zone in id_regions for b in zone2box2p[zone].keys()]))

    logger.info("Splitting full dataset")
    val_boxes, test_boxes, train_boxes = split_boxes(
        id_boxes, lat_lon_date_path, dicts, path2zb, args
    )

    val_paths = [p for b in val_boxes for (p, z) in box2pz[b]]
    test_paths = [p for b in test_boxes for (p, z) in box2pz[b]]
    train_paths = [p for b in train_boxes for (p, z) in box2pz[b]]
    logger.info(
        "Split remaining %s tiles in %s-%s-%s for full train-val-test"
        % (len(id_boxes), len(train_boxes), len(val_boxes), len(test_boxes))
    )

    logger.info("Building maps...")
    ood_map, ood_admin_map = build_maps(
        ood_boxes,
        ood_regions,
        box2pz,
        zone2box2p,
        dicts,
    )
    val_map, val_admin_map = build_maps(
        val_boxes,
        id_regions,
        box2pz,
        zone2box2p,
        dicts,
    )
    train_map, train_admin_map = build_maps(
        train_boxes,
        id_regions,
        box2pz,
        zone2box2p,
        dicts,
    )
    test_map, test_admin_map = build_maps(
        test_boxes,
        id_regions,
        box2pz,
        zone2box2p,
        dicts,
    )

    ood_split_dict = {
        "regions": ood_regions,
        "paths": ood_paths,
        "boxes": ood_boxes,
        "map": ood_map,
        "admin_map": ood_admin_map,
    }
    val_split_dict = {
        "regions": id_regions,
        "boxes": list(val_boxes),
        "paths": val_paths,
        "map": val_map,
        "admin_map": val_admin_map,
    }
    test_split_dict = {
        "regions": id_regions,
        "paths": test_paths,
        "boxes": list(test_boxes),
        "map": test_map,
        "admin_map": test_admin_map,
    }
    train_split_dict = {
        "regions": id_regions,
        "boxes": list(train_boxes),
        "paths": train_paths,
        "map": train_map,
        "admin_map": train_admin_map,
    }
    downstr_split_dict = {
        "val_regions": downstr_val_regions,
        "test_regions": downstr_test_regions,
        "train_regions": downstr_train_regions,
        "ood_regions": ood_regions,
        "paths": ood_paths + val_paths + test_paths + train_paths,
    }

    for name, splits in (
        ("ood_splits", ood_split_dict),
        ("val_splits", val_split_dict),
        ("test_splits", test_split_dict),
        ("train_splits", train_split_dict),
        ("downstr_splits", downstr_split_dict),
    ):
        out_path = os.path.join(out_dir, "%s.json" % name)
        logger.info("Writing splits to %s" % out_path)
        with open(out_path, "w") as f:
            json.dump(splits, f)

    if args.compute_decr_sizes:
        random.seed(42)
        tr = downstr_split_dict["train_regions"]

        regs = tr
        for perc in (100, 70, 50, 20, 5, 1):
            regs = random.sample(regs, k=round(len(tr) * perc / 100.0))
            print(
                "perc: %s, took %s regions out of %s, prop %s"
                % (perc, len(regs), len(tr), len(regs) / len(tr))
            )
            downstr_split_dict[f"train_regions_{perc}"] = regs

        assert set(downstr_split_dict["train_regions"]) == set(
            downstr_split_dict["train_regions_100"]
        )
        assert set(downstr_split_dict["train_regions_70"]).issubset(
            set(downstr_split_dict["train_regions_100"])
        )

        out_path = os.path.join(out_dir, "downstr_splits_incl_small.json")
        logger.info("Writing downstream splits with decreasing sizes to %s" % out_path)
        with open(out_path, "w") as f:
            json.dump(downstr_split_dict, f)


def build_split_maps(splits, dicts, box2pz, zone2box2p, out_dir):
    ood_regions = splits["ood_regions"]
    id_regions = splits["id_regions"]
    ood_boxes = splits["ood_boxes"]
    val_boxes = splits["val_boxes"]
    test_boxes = splits["test_boxes"]
    train_boxes = splits["train_boxes"]
    ood_map, ood_admin_map = build_maps(
        ood_boxes,
        ood_regions,
        box2pz,
        zone2box2p,
        dicts,
    )
    val_map, val_admin_map = build_maps(
        val_boxes,
        id_regions,
        box2pz,
        zone2box2p,
        dicts,
    )
    train_map, train_admin_map = build_maps(
        train_boxes,
        id_regions,
        box2pz,
        zone2box2p,
        dicts,
    )
    test_map, test_admin_map = build_maps(
        test_boxes,
        id_regions,
        box2pz,
        zone2box2p,
        dicts,
    )
    valtest_map, valtest_admin_map = build_maps(
        splits["val_boxes"] + splits["test_boxes"],
        splits["id_regions"],
        box2pz,
        zone2box2p,
        dicts,
    )
    split_maps = {
        "ood_map": ood_map,
        "ood_admin_map": ood_admin_map,
        "val_map": val_map,
        "val_admin_map": val_admin_map,
        "train_map": train_map,
        "train_admin_map": train_admin_map,
        "test_map": test_map,
        "test_admin_map": test_admin_map,
        "valtest_map": valtest_map,
        "valtest_admin_map": valtest_admin_map,
    }
    out_path = os.path.join(out_dir, "split_maps.json")
    logger.info("Writing split maps to %s" % out_path)
    with open(out_path, "w") as f:
        json.dump(split_maps, f)
    return split_maps


def load_indices(args):
    logger.info("Loading indices from %s" % args.in_dir)
    with open(os.path.join(args.in_dir, "tilepath_2_adminzone_box.json"), "r") as f:
        path2zb = json.load(f)
    with open(os.path.join(args.in_dir, "adminzone_2_box_2_tilepath.json"), "r") as f:
        zone2box2p = json.load(f)
    with open(os.path.join(args.in_dir, "box_2_tilepath_adminzone.json"), "r") as f:
        box2pz = json.load(f)
    with open(os.path.join(args.in_dir, "out_of_region_tiles.json"), "r") as f:
        none_tiles = json.load(f)
    return path2zb, zone2box2p, box2pz, none_tiles


def load_maps(args):
    logger.info("Loading maps from %s" % args.in_dir)
    with open(os.path.join(args.in_dir, "maps.json"), "r") as f:
        maps = json.load(f)
    logger.info("Loading dicts from %s" % args.in_dir)
    with open(os.path.join(args.in_dir, "dicts.json"), "r") as f:
        dicts = json.load(f)
    return dicts, maps


def run_preprocess(args):
    if args.recompute_indices:
        path2zb, zone2box2p, box2pz, none_tiles = compute_admin_zones_coords(
            args.tiles_path,
            args.admin_shapes,
            args.json_out_path,
            args.num_workers,
            debug=args.debug,
            remove=args.remove,
        )
    else:
        path2zb, zone2box2p, box2pz, none_tiles = load_indices(args)

    if args.recompute_maps:
        dicts, maps = build_dicts(
            path2zb, zone2box2p, box2pz, none_tiles, args.json_out_path
        )
    else:
        dicts, maps = load_maps(args)

    if args.recompute_splits:
        train_val_test_splits(
            zone2box2p,
            box2pz,
            path2zb,
            maps["lat_lon_date_path"],
            dicts,
            args.json_out_path,
            args,
        )

    logger.info("Done")


def define_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--admin_shapes", type=str, default="data/SO_Admin2_1990/SO_Admin2_1990.shp"
    )
    parser.add_argument(
        "--tiles_path",
        type=str,
        default="/path/to/landsat8/somalia/tiles/",  # todo
    )
    parser.add_argument("--json_out_path", type=str, default="data/indices/")
    parser.add_argument("--in_dir", type=str, default="data/indices/")
    parser.add_argument("--recompute_indices", action="store_true")
    parser.add_argument("--recompute_maps", action="store_true")
    parser.add_argument("--recompute_splits", action="store_true")
    parser.add_argument("--compute_decr_sizes", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # remove tiles that have no overlap with the provided shapefile
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--ood_regions", type=int, default=4)
    parser.add_argument("--val_downstr", type=int, default=7)
    parser.add_argument("--test_downstr", type=int, default=7)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--test_fraction", type=float, default=0.0)
    parser.add_argument(
        "--distances", type=tuple, default=(0.15, 0.4)
    )  # in degrees lat/lon
    parser.add_argument(
        "--per_distance", type=tuple, default=(8, 48)
    )  # in degrees lat/lon
    return parser


if __name__ == "__main__":
    """
    Set the path to the directory where the tiles are saved with `--tiles_path` or above.
    First time, run as
    `python scripts/preprocess/build_indices.py --recompute_indices --recompute_maps --recompute_splits --tiles_path <path_to_tiles>`
        (add `--compute_decr_sizes` if you want to store IPC prediction training sets with decreasing size)
    """
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    parser = define_args()
    args = parser.parse_args()
    random.seed(args.seed)

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = "output/%s_preprocess" % timestamp

    utils.mkdir_p(output_dir)
    utils.mkdir_p(args.json_out_path)
    utils.initialize_logging(output_dir, to_file=True, logger_name="__main__")
    logger.info(vars(args))

    run_preprocess(args)

    """
    with open(os.path.join(args.in_dir, 'tilepath_2_adminzone_box.json'), 'r') as f:
        path2zb = json.load(f)
    with open(os.path.join(args.in_dir, 'adminzone_2_box_2_tilepath.json'), 'r') as f:
        zone2box2p = json.load(f)
    with open(os.path.join(args.in_dir, 'box_2_tilepath_adminzone.json'), 'r') as f:
        box2pz = json.load(f)
    with open(os.path.join(args.in_dir, 'out_of_region_tiles.json'), 'r') as f:
        none_tiles = json.load(f)
    """
