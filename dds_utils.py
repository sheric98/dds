import re
import os
import csv
import rectpack
import shutil
import subprocess
import numpy as np
import cv2 as cv
import networkx
from networkx.algorithms.components.connected import connected_components


class ServerConfig:
    def __init__(self, low_res, high_res, low_qp, high_qp, bsize,
                 h_thres, l_thres, max_obj_size, min_obj_size,
                 tracker_length, boundary, intersection_threshold,
                 tracking_threshold, suppression_threshold, simulation,
                 rpn_enlarge_ratio, prune_score, objfilter_iou, size_obj):
        self.low_resolution = low_res
        self.high_resolution = high_res
        self.low_qp = low_qp
        self.high_qp = high_qp
        self.batch_size = bsize
        self.high_threshold = h_thres
        self.low_threshold = l_thres
        self.max_object_size = max_obj_size
        self.min_object_size = min_obj_size
        self.tracker_length = tracker_length
        self.boundary = boundary
        self.intersection_threshold = intersection_threshold
        self.simulation = simulation
        self.tracking_threshold = tracking_threshold
        self.suppression_threshold = suppression_threshold
        self.rpn_enlarge_ratio = rpn_enlarge_ratio
        self.prune_score = prune_score
        self.objfilter_iou = objfilter_iou
        self.size_obj = size_obj


class Region:
    def __init__(self, fid, x, y, w, h, conf, label, resolution,
                 origin="generic"):
        self.fid = int(fid)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.conf = float(conf)
        self.label = label
        self.resolution = float(resolution)
        self.origin = origin

    @staticmethod
    def convert_from_server_response(r, res, phase):
        return Region(r[0], r[1], r[2], r[3], r[4], r[5], r[6], res, phase)

    def __str__(self):
        string_rep = (f"{self.fid}, {self.x:0.3f}, {self.y:0.3f}, "
                      f"{self.w:0.3f}, {self.h:0.3f}, {self.conf:0.3f}, "
                      f"{self.label}, {self.origin}")
        return string_rep

    # these two are just so I can use regions as keys in dict
    def __hash__(self):
        return hash((self.fid, self.x, self.y, self.w, self.h))

    def __eq__(self, other):
        return (self.fid, self.x, self.y, self.w, self.h) == (other.fix, other.x, other.y, other.w, other.h)

    def is_same(self, region_to_check, threshold=0.5):
        # If the fids or labels are different
        # then not the same
        if (self.fid != region_to_check.fid or
                ((self.label != "-1" and region_to_check.label != "-1") and
                 (self.label != region_to_check.label))):
            return False

        # If the intersection to union area
        # ratio is greater than the threshold
        # then the regions are the same
        if calc_iou(self, region_to_check) > threshold:
            return True
        else:
            return False

    def enlarge(self, ratio):
        x_min = max(self.x - self.w * ratio, 0.0)
        y_min = max(self.y - self.h * ratio, 0.0)
        x_max = min(self.x + self.w * (1 + ratio), 1.0)
        y_max = min(self.y + self.h * (1 + ratio), 1.0)
        self.x = x_min
        self.y = y_min
        self.w = x_max - x_min
        self.h = y_max - y_min

    def copy(self):
        return Region(self.fid, self.x, self.y, self.w, self.h, self.conf,
                      self.label, self.resolution, self.origin)


class Results:
    def __init__(self):
        self.regions = []
        self.regions_dict = {}

    def __len__(self):
        return len(self.regions)

    def results_high_len(self, threshold):
        count = 0
        for r in self.regions:
            if r.conf > threshold:
                count += 1
        return count

    def is_dup(self, result_to_add, threshold=0.5):
        # return the regions with IOU greater than threshold
        # and maximum confidence
        if result_to_add.fid not in self.regions_dict:
            return None

        max_conf = -1
        max_conf_result = None
        for existing_result in self.regions_dict[result_to_add.fid]:
            if existing_result.is_same(result_to_add, threshold):
                if existing_result.conf > max_conf:
                    max_conf = existing_result.conf
                    max_conf_result = existing_result
        return max_conf_result

    def combine_results(self, additional_results, threshold=0.5):
        for result_to_add in additional_results.regions:
            self.add_single_result(result_to_add, threshold)

    def add_single_result(self, region_to_add, threshold=0.5):
        if threshold == 1:
            self.append(region_to_add)
            return
        dup_region = self.is_dup(region_to_add, threshold)
        if (not dup_region or
                ("tracking" in region_to_add.origin and
                 "tracking" in dup_region.origin)):
            self.regions.append(region_to_add)
            if region_to_add.fid not in self.regions_dict:
                self.regions_dict[region_to_add.fid] = []
            self.regions_dict[region_to_add.fid].append(region_to_add)
        else:
            final_object = None
            if dup_region.origin == region_to_add.origin:
                final_object = max([region_to_add, dup_region],
                                   key=lambda r: r.conf)
            elif ("low" in dup_region.origin and
                  "high" in region_to_add.origin):
                final_object = region_to_add
            elif ("high" in dup_region.origin and
                  "low" in region_to_add.origin):
                final_object = dup_region
            else:
                if region_to_add.conf > dup_region.conf:
                    final_object = region_to_add
                else:
                    final_object = dup_region
            dup_region.x = final_object.x
            dup_region.y = final_object.y
            dup_region.w = final_object.w
            dup_region.h = final_object.h
            dup_region.conf = final_object.conf
            dup_region.origin = final_object.origin

    def suppress(self, threshold=0.5):
        new_regions_list = []
        while len(self.regions) > 0:
            max_conf_obj = max(self.regions, key=lambda e: e.conf)
            new_regions_list.append(max_conf_obj)
            self.remove(max_conf_obj)
            objs_to_remove = []
            for r in self.regions:
                if r.fid != max_conf_obj.fid:
                    continue
                if calc_iou(r, max_conf_obj) > threshold:
                    objs_to_remove.append(r)
            for r in objs_to_remove:
                self.remove(r)
        new_regions_list.sort(key=lambda e: e.fid)
        for r in new_regions_list:
            self.append(r)

    def append(self, region_to_add):
        self.regions.append(region_to_add)
        if region_to_add.fid not in self.regions_dict:
            self.regions_dict[region_to_add.fid] = []
        self.regions_dict[region_to_add.fid].append(region_to_add)

    def remove(self, region_to_remove):
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)
        self.regions.remove(region_to_remove)
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)

    def fill_gaps(self, number_of_frames):
        if len(self.regions) == 0:
            return
        results_to_add = Results()
        max_resolution = max([e.resolution for e in self.regions])
        fids_in_results = [e.fid for e in self.regions]
        for i in range(number_of_frames):
            if i not in fids_in_results:
                results_to_add.regions.append(Region(i, 0, 0, 0, 0,
                                                     0.1, "no obj",
                                                     max_resolution))
        self.combine_results(results_to_add)
        self.regions.sort(key=lambda r: r.fid)

    def write_results_txt(self, fname):
        results_file = open(fname, "w")
        for region in self.regions:
            # prepare the string to write
            str_to_write = (f"{region.fid},{region.x},{region.y},"
                            f"{region.w},{region.h},"
                            f"{region.label},{region.conf},"
                            f"{region.resolution},{region.origin}\n")
            results_file.write(str_to_write)
        results_file.close()

    def write_results_csv(self, fname):
        results_files = open(fname, "w")
        csv_writer = csv.writer(results_files)
        for region in self.regions:
            row = [region.fid, region.x, region.y,
                   region.w, region.h,
                   region.label, region.conf,
                   region.resolution, region.origin]
            csv_writer.writerow(row)
        results_files.close()

    def write(self, fname):
        if re.match(r"\w+[.]csv\Z", fname):
            self.write_results_csv(fname)
        else:
            self.write_results_txt(fname)

class MoveResults(Results):
    def __init__(self):
        super().__init__()
        self.move_to_origs = {}
    
    def append(self, region, orig_fid):
        super().append(region)
        fid = region.fid
        if fid not in self.move_to_origs:
            self.move_to_origs[fid] = set()
        self.move_to_origs[fid].add(orig_fid)


class BBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def filter_bbox_group(bb1, bb2, iou_threshold):
    if calc_iou(bb1, bb2) > iou_threshold and bb1.label == bb2.label:
        return True
    else:
        return False


def overlap(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1.x, bb2.x)
    y_top = max(bb1.y, bb2.y)
    x_right = min(bb1.x+bb1.w, bb2.x+bb2.w)
    y_bottom = min(bb1.y+bb1.h, bb2.y+bb2.h)

    # no overlap
    if x_right < x_left or y_bottom < y_top:
        return False
    else:
        return True


def pairwise_overlap_indexing_list(single_result_frame, iou_threshold):
    pointwise = [[i] for i in range(len(single_result_frame))]
    pairwise = [[i, j] for i, x in enumerate(single_result_frame)
                for j, y in enumerate(single_result_frame)
                if i != j if filter_bbox_group(x, y, iou_threshold)]
    return pointwise + pairwise


def simple_merge(single_result_frame, index_to_merge):
    # directly using the largest box
    bbox_large = []
    for i in index_to_merge:
        i2np = np.array([j for j in i])
        left = min(np.array(single_result_frame)[i2np], key=lambda x: x.x)
        top = min(np.array(single_result_frame)[i2np], key=lambda x: x.y)
        right = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.x + x.w)
        bottom = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.y + x.h)

        fid, x, y, w, h, conf, label, resolution, origin = (
            left.fid, left.x, top.y, right.x + right.w - left.x,
            bottom.y + bottom.h - top.y, left.conf, left.label,
            left.resolution, left.origin)
        single_merged_region = Region(fid, x, y, w, h, conf,
                                      label, resolution, origin)
        bbox_large.append(single_merged_region)
    return bbox_large


def merge_boxes_in_results(results_dict, min_conf_threshold, iou_threshold):
    final_results = Results()

    # Clean dict to remove min_conf_threshold
    for _, regions in results_dict.items():
        to_remove = []
        for r in regions:
            if r.conf < min_conf_threshold:
                to_remove.append(r)
        for r in to_remove:
            regions.remove(r)

    for fid, regions in results_dict.items():
        overlap_pairwise_list = pairwise_overlap_indexing_list(
            regions, iou_threshold)
        overlap_graph = to_graph(overlap_pairwise_list)
        grouped_bbox_idx = [c for c in sorted(
            connected_components(overlap_graph), key=len, reverse=True)]
        merged_regions = simple_merge(regions, grouped_bbox_idx)
        for r in merged_regions:
            final_results.append(r)
    return final_results


def read_results_csv_dict(fname):
    """Return a dictionary with fid mapped to an array
    that contains all Regions objects"""
    results_dict = {}

    rows = []
    with open(fname) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            rows.append(row)

    for row in rows:
        fid = int(row[0])
        x, y, w, h = [float(e) for e in row[1:5]]
        conf = float(row[6])
        label = row[5]
        resolution = float(row[7])
        origin = float(row[8])

        region = Region(fid, x, y, w, h, conf, label, resolution, origin)

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(region)

    return results_dict


def read_results_txt_dict(fname):
    """Return a dictionary with fid mapped to
       and array that contains all SingleResult objects
       from that particular frame"""
    results_dict = {}

    with open(fname, "r") as f:
        lines = f.readlines()
        f.close()

    for line in lines:
        line = line.split(",")
        fid = int(line[0])
        x, y, w, h = [float(e) for e in line[1:5]]
        conf = float(line[6])
        label = line[5]
        resolution = float(line[7])
        origin = "generic"
        if len(line) > 8:
            origin = line[8].strip()
        single_result = Region(fid, x, y, w, h, conf, label,
                               resolution, origin.rstrip())

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(single_result)

    return results_dict


def read_results_dict(fname):
    # TODO: Need to implement a CSV function
    if re.match(r"\w+[.]csv\Z", fname):
        return read_results_csv_dict(fname)
    else:
        return read_results_txt_dict(fname)


def calc_intersection_area(a, b):
    to = max(a.y, b.y)
    le = max(a.x, b.x)
    bo = min(a.y + a.h, b.y + b.h)
    ri = min(a.x + a.w, b.x + b.w)

    w = max(0, ri - le)
    h = max(0, bo - to)

    return w * h


def calc_area(a):
    w = max(0, a.w)
    h = max(0, a.h)

    return w * h


def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area
    if union_area == 0:
        return float('inf')
    return intersection_area / union_area


def get_interval_area(width, all_yes):
    area = 0
    for y1, y2 in all_yes:
        area += (y2 - y1) * width
    return area


def insert_range_y(all_yes, y1, y2):
    ranges_length = len(all_yes)
    idx = 0
    while idx < ranges_length:
        if not (y1 > all_yes[idx][1] or all_yes[idx][0] > y2):
            # Overlapping
            y1 = min(y1, all_yes[idx][0])
            y2 = max(y2, all_yes[idx][1])
            del all_yes[idx]
            ranges_length = len(all_yes)
        else:
            idx += 1

    all_yes.append((y1, y2))


def get_y_ranges(regions, j, x1, x2):
    all_yes = []
    while j < len(regions):
        if (x1 < (regions[j].x + regions[j].w) and
                x2 > regions[j].x):
            y1 = regions[j].y
            y2 = regions[j].y + regions[j].h
            insert_range_y(all_yes, y1, y2)
        j += 1
    return all_yes


def compute_area_of_frame(regions):
    regions.sort(key=lambda r: r.x + r.w)

    all_xes = []
    for r in regions:
        all_xes.append(r.x)
        all_xes.append(r.x + r.w)
    all_xes.sort()

    area = 0
    j = 0
    for i in range(len(all_xes) - 1):
        x1 = all_xes[i]
        x2 = all_xes[i + 1]

        if x1 < x2:
            while (regions[j].x + regions[j].w) < x1:
                j += 1
            all_yes = get_y_ranges(regions, j, x1, x2)
            area += get_interval_area(x2 - x1, all_yes)

    return area


def compute_area_of_regions(results):
    if len(results.regions) == 0:
        return 0

    min_frame = min([r.fid for r in results.regions])
    max_frame = max([r.fid for r in results.regions])

    total_area = 0
    for fid in range(min_frame, max_frame + 1):
        regions_for_frame = [r for r in results.regions if r.fid == fid]
        total_area += compute_area_of_frame(regions_for_frame)

    return total_area


def compress_and_get_size(images_path, start_id, end_id, qp,
                          enforce_iframes=False, resolution=None):
    number_of_frames = end_id - start_id
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    if resolution and enforce_iframes:
        scale = f"scale=trunc(iw*{resolution}/2)*2:trunc(ih*{resolution}/2)*2"
        if not qp:
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{images_path}/%010d.png",
                                              "-vcodec", "libx264", "-g", "15",
                                              "-keyint_min", "15",
                                              "-pix_fmt", "yuv420p",
                                              "-vf", scale,
                                              "-frames:v",
                                              str(number_of_frames),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
        else:
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{images_path}/%010d.png",
                                              "-vcodec", "libx264",
                                              "-g", "15",
                                              "-keyint_min", "15",
                                              "-qp", f"{qp}",
                                              "-pix_fmt", "yuv420p",
                                              "-vf", scale,
                                              "-frames:v",
                                              str(number_of_frames),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
    else:
        encoding_result = subprocess.run(["ffmpeg", "-y",
                                          "-start_number", str(start_id),
                                          "-i", f"{images_path}/%010d.png",
                                          "-loglevel", "error",
                                          "-vcodec", "libx264",
                                          "-pix_fmt", "yuv420p", "-crf", "23",
                                          encoded_vid_path],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         universal_newlines=True)

    size = 0
    if encoding_result.returncode != 0:
        # Encoding failed
        print("ENCODING FAILED")
        print(encoding_result.stdout)
        print(encoding_result.stderr)
        exit()
    else:
        size = os.path.getsize(encoded_vid_path)

    return size


def extract_images_from_video(images_path, req_regions, move_to_orig=None):
    if not os.path.isdir(images_path):
        return

    for fname in os.listdir(images_path):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path, fname))
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    extacted_images_path = os.path.join(images_path, "%010d.png")
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",
                                      "-g", "8", "-q:v", "2",
                                      "-vsync", "0", "-start_number", "0",
                                      extacted_images_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()

    fnames = sorted(
        [os.path.join(images_path, name)
            for name in os.listdir(images_path) if "png" in name])
    
    if move_to_orig is None:
        fids = sorted(list(set([r.fid for r in req_regions.regions])))
        fids_mapping = zip(fids, fnames)

    else:
        move_to_orig_fids = {}

        for region in req_regions.regions:
            fid = str(region.fid)
            orig_fid = move_to_orig[region].fid
            if fid not in move_to_orig_fids:
                move_to_orig_fids[fid] = set()
            move_to_orig_fids[fid].add(orig_fid)
        
        for key, item in move_to_orig_fids.items():
            orig_list = sorted([orig_fid for orig_fid in item])
            orig_str_list = list(map(lambda x: str(x), orig_list))
            png_name = '_'.join(orig_str_list)

            move_to_orig_fids[key] = png_name

        pairs = list(move_to_orig_fids.items())
        pairs.sort(key=lambda x: int(x[0]))
        fids_mapping = zip(pairs, fnames)

    for fname in fnames:
        # Rename temporarily
        os.rename(fname, f"{fname}_temp")

    for x, fname in fids_mapping:
        if move_to_orig is None:
            os.rename(os.path.join(f"{fname}_temp"),
                      os.path.join(images_path, f"{str(x).zfill(10)}.png"))
        else:
            os.rename(os.path.join(f"{fname}_temp"),
                      os.path.join(images_path, f"{x[0] + '_' + x[1]}.png"))

def crop_images(results, vid_name, images_direc, resolution=None,
                orig_to_move=None, normalize=False):
    cached_image = None
    cropped_images = {}

    for region in results.regions:
        if not (cached_image and
                cached_image[0] == region.fid):
            image_path = os.path.join(images_direc,
                                      f"{str(region.fid).zfill(10)}.png")
            cached_image = (region.fid, cv.imread(image_path))

        if orig_to_move is None:
            f_idx = region.fid
            r_x = region.x
            r_y = region.y
        else:
            f_idx = orig_to_move[region].fid
            r_x = orig_to_move[region].x
            r_y = orig_to_move[region].y
        
        # Just move the complete image
        if r_x == 0 and r_y == 0 and region.w == 1 and region.h == 1:
            cropped_images[f_idx] = cached_image[1]
            continue

        width = cached_image[1].shape[1]
        height = cached_image[1].shape[0]
        x0_move = int(r_x * width)
        y0_move = int(r_y * height)
        x1_move = int((region.w * width) + x0_move - 1)
        y1_move = int((region.h * height) + y0_move - 1)
        x0_orig = int(region.x * width)
        y0_orig = int(region.y * height)
        x1_orig = int((region.w * width) + x0_orig - 1)
        y1_orig = int((region.h * height) + y0_orig - 1)

        base = [.485, .456, .406]
        if f_idx not in cropped_images:
            base_im = np.zeros_like(cached_image[1])
            if normalize:
                base_im[:,:,0] = base[0]*255
                base_im[:,:,1] = base[1]*255
                base_im[:,:,2] = base[2]*255

            cropped_images[f_idx] = base_im


        cropped_image = cropped_images[f_idx]
        cropped_image[y0_move:y1_move, x0_move:x1_move, :] = cached_image[1][y0_orig:y1_orig, x0_orig:x1_orig, :]
        cropped_images[f_idx] = cropped_image

    os.makedirs(vid_name, exist_ok=True)
    frames_count = len(cropped_images)
    frames = sorted(cropped_images.items(), key=lambda e: e[0])
    for idx, (_, frame) in enumerate(frames):
        if resolution:
            w = int(frame.shape[1] * resolution)
            h = int(frame.shape[0] * resolution)
            im_to_write = cv.resize(frame, (w, h), fx=0, fy=0,
                                    interpolation=cv.INTER_CUBIC)
            frame = im_to_write
        cv.imwrite(os.path.join(vid_name, f"{str(idx).zfill(10)}.png"), frame,
                   [cv.IMWRITE_PNG_COMPRESSION, 0])

    return frames_count


def merge_images(cropped_images_direc, low_images_direc, move_regions,
                 move_to_orig, context, normalize, debug_mode, start_idx, end_idx):

    if debug_mode:
        os.makedirs("debugging", exist_ok=True)
        vid_name = cropped_images_direc.split('/')[1].split('-')[0]
        debug_folder = os.path.join("debugging", f'{vid_name}-merged')
        os.makedirs(debug_folder, exist_ok=True)
        debug_path = os.path.join(debug_folder, f'{vid_name}-{start_idx}-{end_idx}-merged')
        os.makedirs(debug_path, exist_ok=True)

    images = {}
    for fid, orig_fids in move_regions.move_to_origs.items():
        
        high_images_dict = {}
        height = -1
        width = -1
        col_range = -1
        for orig_fid in orig_fids:
            fid_name = f'{str(orig_fid).zfill(10)}.png'
            high_image = cv.imread(os.path.join(cropped_images_direc, fid_name))
            high_images_dict[orig_fid] = high_image

            # set max values
            height = max(height, high_image.shape[0])
            width = max(width, high_image.shape[1])
            col_range = max(col_range, high_image.shape[2])

        base = [.485, .456, .406]
        base_im = np.zeros((height, width, col_range))
        if normalize:
            base_im[:,:,0] = base[0]*255
            base_im[:,:,1] = base[1]*255
            base_im[:,:,2] = base[2]*255

        for orig_fid, high_im in high_images_dict.items():
            if high_im.shape[0] != height or high_im.shape[1] != width:
                enlarge_high = cv.resize(high_im, (width, height), fx=0, fy=0,
                                         interpolation=cv.INTER_CUBIC)
                high_images_dict[orig_fid] = enlarge_high

        # Read low resolution image
        low_images_dict = {}
        for orig_fid in orig_fids:
            fid_name = f'{str(orig_fid).zfill(10)}.png'
            low_image = cv.imread(os.path.join(low_images_direc, fid_name))
            # enlarge low resolution image
            enlarged_image = cv.resize(low_image, (width, height), fx=0, fy=0,
                                       interpolation=cv.INTER_CUBIC)
            low_images_dict[orig_fid] = enlarged_image

        # Put regions in place
        regions = move_regions.regions_dict[fid]
        high_coords = []
        for r in regions:
            orig_r = move_to_orig[r]
            low_image = low_images_dict[orig_r.fid]
            high_image = high_images_dict[orig_r.fid]

            c = []
            high_c = []

            for region in [orig_r, r]:

                context_x = max(region.x - context, 0)
                context_y = max(region.y - context, 0)
                context_x_end = min(region.x + region.w + context, 1)
                context_y_end = min(region.y + region.h + context, 1)

                context_x0 = int(context_x * width)
                context_y0 = int(context_y * height)
                context_x1 = int(context_x_end * width - 1)
                context_y1 = int(context_y_end * height - 1)

                c.append([context_x0, context_x1, context_y0, context_y1])

                x0 = int(region.x * width)
                y0 = int(region.y * height)
                x1 = int((region.w * width) + x0 - 1)
                y1 = int((region.h * height) + y0 - 1)
                high_c.append([x0, y0, x1, y1])
            
            high_c.append(orig_r.fid)
            
            # make sure widths and heights are the same
            orig_w = c[0][1] - c[0][0]
            r_w = c[1][1] - c[1][0]
            orig_h = c[0][3] - c[0][2]
            r_h = c[1][3] - c[1][2]

            # also adjust left alignment / right alignment
            if orig_w < r_w:
                c[1][1] = c[1][0] + orig_w
                if orig_r.x - context < 0:
                    c[1][1] += int(abs(orig_r.x - context) * width)
                    c[1][0] += int(abs(orig_r.x - context) * width)
            elif r_w < orig_w:
                c[0][1] = c[0][0] + r_w
            if orig_h < r_h:
                c[1][3] = c[1][2] + orig_h
                if orig_r.y - context < 0:
                    c[1][3] += int(abs(orig_r.y - context) * height)
                    c[1][2] += int(abs(orig_r.y - context) * height)
            elif r_h < orig_h:
                c[0][3] = c[0][2] + r_h

            high_coords.append(high_c)

            # overlay low-res images first
            base_im[c[1][2]:c[1][3], c[1][0]:c[1][1], :] =\
                low_image[c[0][2]:c[0][3], c[0][0]:c[0][1], :]

        # put high-res on top
        for coords in high_coords:
            x0, y0, x1, y1 = coords[0]
            mov_x0, mov_y0, mov_x1, mov_y1 = coords[1]
            high_image = high_images_dict[coords[2]]
            base_im[mov_y0:mov_y1, mov_x0:mov_x1, :] = high_image[y0:y1, x0:x1, :]
        
        # save image
        sorted_origs = sorted(list(orig_fids))
        orig_str = '_'.join([str(orig_fid) for orig_fid in sorted_origs])
        fname = f'{str(fid)}_{orig_str}.png'

        img_path = os.path.join(cropped_images_direc, fname)
        cv.imwrite(img_path, base_im,
                   [cv.IMWRITE_PNG_COMPRESSION, 0])
        if debug_mode:
            shutil.copyfile(img_path, os.path.join(debug_path, fname))
        
        images[fid] = cv.imread(img_path)
    return images


def combine_regions_map(results, padding=0, context=0, grouping=2):
    DEC_PLACES = 20
    ENTIRE_FRAME = rectpack.float2dec(1, DEC_PLACES)
    
    frame_regions = list(results.regions_dict.items())
    frame_regions.sort(key=lambda x: x[0])
    frame_regions = list(zip(*frame_regions))[1]

    orig_to_move = {}
    move_to_orig = {}
    move_regions = MoveResults()

    new_fid = 0

    if not grouping:
        grouping = len(frame_regions)

    for i in range(0, len(frame_regions), grouping):
    
        to_combine = frame_regions[i:i+grouping]
        
        # don't move anything if only one frame
        if len(to_combine) == 1:
            for region in to_combine[0]:
                new_region = region.copy()
                new_region.fid = new_fid
                orig_to_move[region] = new_region
                move_to_orig[new_region] = region
                move_regions.append(new_region, region.fid)
                #print(str(new_region))
            new_fid += 1

        else:
            combine_regions = []
            for regions in to_combine:
                combine_regions.extend(regions)
            
            dec_rects = []
            for r in combine_regions:
                pad_w = r.w + 2*(padding + context)
                pad_h = r.h + 2*(padding + context)
                if pad_w > 1:
                    pad_w = 1
                if pad_h > 1:
                    pad_h = 1
                rect_tuple = (rectpack.float2dec(pad_w, DEC_PLACES), rectpack.float2dec(pad_h, DEC_PLACES))
                dec_rects.append(rect_tuple)

            packer = rectpack.newPacker(rotation=False)
            for idx, r in enumerate(dec_rects):
                packer.add_rect(*r, rid=idx)
            packer.add_bin(ENTIRE_FRAME, ENTIRE_FRAME, count=float('inf'))

            packer.pack()

            # process packed rectangles
            
            all_rects = packer.rect_list()
            for rect in all_rects:
                b, x, y, w, h, rid = rect

                float_x = float(x / ENTIRE_FRAME) + padding + context
                float_y = float(y / ENTIRE_FRAME) + padding + context
            
                orig_region = combine_regions[rid]

                new_region = orig_region.copy()
                new_region.x = float_x
                new_region.y = float_y
                new_region.fid = new_fid + b

                #print(str(new_region))

                orig_to_move[orig_region] = new_region
                move_to_orig[new_region] = orig_region
                move_regions.append(new_region, orig_region.fid)

            #print('Combined into %d frames' % len(packer))

            new_fid += len(packer)

    return orig_to_move, move_to_orig, move_regions


def move_region_to_orig(res_region, move_region, orig_region):
    """ returns res region mapped to original position
    """
    new_x = orig_region.x + res_region.x - move_region.x
    new_y = orig_region.y + res_region.y - move_region.y
    new_res = res_region.copy()
    new_res.fid = orig_region.fid
    new_res.x = new_x
    new_res.y = new_y

    return new_res


def check_res_overlap(res_region, check_region, context):
    bbox_x = check_region.x - context
    bbox_y = check_region.y - context
    bbox_w = check_region.w + 2 * context
    bbox_h = check_region.h + 2 * context
    check_bbox = BBox(bbox_x, bbox_y, bbox_w, bbox_h)

    return calc_iou(res_region, check_bbox)


def crop_matched_region(overlap_region, context, res_region):
    new_res = res_region.copy()
    new_res.x = max(res_region.x, overlap_region.x - context)
    new_res.y = max(res_region.y, overlap_region.y - context)
    new_res.w = min(res_region.x + res_region.w, overlap_region.x + overlap_region.w + context) - new_res.x
    new_res.h = min(res_region.y + res_region.h, overlap_region.y + overlap_region.h + context) - new_res.y

    return new_res


def convert_move_results(move_results, move_regions, move_to_orig, padding, context,
                         iou_thresh, reduced):
    orig_results = Results()
    orig_bb_to_move = {}

    for res_region in move_results.regions:
        check_regions = move_regions.regions_dict[res_region.fid]
        overlap_region = None
        for check_region in check_regions:
            # enough intersect
            if check_res_overlap(res_region, check_region, context) > iou_thresh:
                if overlap_region is None:
                    overlap_region = check_region
                else:
                    overlap_region = None
                    break
        if overlap_region is not None:
            final_res_region = res_region
            if reduced:
                final_res_region = crop_matched_region(overlap_region, context, res_region)
            orig_region = move_to_orig[overlap_region]
            orig_res_region = move_region_to_orig(final_res_region, overlap_region, orig_region)

            orig_bb = (orig_res_region.x, orig_res_region.y, orig_res_region.w,
                       orig_res_region.h, orig_res_region.label, orig_res_region.conf,
                       orig_res_region.fid)
            orig_bb_to_move[orig_bb] = final_res_region
            
            orig_results.add_single_result(orig_res_region)

    return orig_results, orig_bb_to_move


def compute_regions_size(results, vid_name, images_direc, resolution, qp,
                         enforce_iframes, estimate_banwidth=True,
                         orig_to_move=None, normalize_crop=False):
    if estimate_banwidth:
        # If not simulation, compress and encode images
        # and get size
        vid_name = f"{vid_name}-cropped"
        frames_count = crop_images(results, vid_name, images_direc,
                                   resolution, orig_to_move, normalize_crop)

        size = compress_and_get_size(vid_name, 0, frames_count, qp=qp,
                                     enforce_iframes=enforce_iframes,
                                     resolution=1)
        pixel_size = compute_area_of_regions(results)
        return size, pixel_size
    else:
        size = compute_area_of_regions(results)

        return size


def draw_move_boxes(move_results, vid_name, start_id, end_id):
    os.makedirs("debugging", exist_ok=True)
    vid_name_end = vid_name.split('/')[-1]
    res_folder = os.path.join('debugging', f'{vid_name_end}-merged')
    os.makedirs(res_folder, exist_ok=True)
    res_path = os.path.join(res_folder, f'{vid_name_end}-{start_id}-{end_id}-merged')
    os.makedirs(res_path, exist_ok=True)
    visualize_regions(move_results, res_path, save=True, high=True)


def draw_move_stats_boxes(tp_bb, fp_bb, fn_bb, vid_name, orig_bb_map, orig_map, context):
    os.makedirs("debugging", exist_ok=True)
    vid_name_end = vid_name.split('/')[-1]
    res_folder = os.path.join('debugging', f'{vid_name_end}-merged')
    os.makedirs(res_folder, exist_ok=True)
    visualize_move_stats(tp_bb, fp_bb, fn_bb, res_folder, orig_bb_map, orig_map, context)


def draw_stats_boxes(tp_bb, fp_bb, fn_bb, vid_name):
    os.makedirs("debugging", exist_ok=True)
    vid_name_end = vid_name.split('/')[-1]
    orig_vid_name = vid_name.split('_dds')[0].split('/')[1]
    orig_vid_path = os.path.join('..', '..', 'videos', orig_vid_name, 'src')
    orig_save_path = os.path.join('debugging', f'{vid_name_end}-stats')
    os.makedirs(orig_save_path, exist_ok=True)
    visualize_stats(tp_bb, fp_bb, fn_bb, orig_vid_path, orig_save_path)


def cleanup(vid_name, debug_mode=False, start_id=None, end_id=None):
    if not os.path.isdir(vid_name + "-cropped"):
        return

    if not debug_mode:
        shutil.rmtree(vid_name + "-base-phase-cropped")
        shutil.rmtree(vid_name + "-cropped")
    else:
        # if start_id is None or end_id is None:
        #     print("Need start_fid and end_fid for debugging mode")
        #     exit()
        # os.makedirs("debugging", exist_ok=True)
        # leaf_direc = vid_name.split("/")[-1] + "-cropped"
        # shutil.move(vid_name + "-cropped", "debugging")
        # shutil.move(os.path.join("debugging", leaf_direc),
        #             os.path.join("debugging",
        #                          f"{leaf_direc}-{start_id}-{end_id}"),
        #             copy_function=os.rename)
        return


def get_size_from_mpeg_results(results_log_path, images_path, resolution):
    with open(results_log_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if line.rstrip().lstrip() != ""]

    num_frames = len([x for x in os.listdir(images_path) if "png" in x])

    bandwidth = 0
    for idx, line in enumerate(lines):
        if f"RES {resolution}" in line:
            bandwidth = float(lines[idx + 2])
            break
    size = bandwidth * 1024.0 * (num_frames / 10.0)
    return size


def filter_results(bboxes, gt_flag, gt_confid_thresh, mpeg_confid_thresh,
                   max_area_thresh_gt, max_area_thresh_mpeg):
    relevant_classes = ["vehicle"]
    if gt_flag:
        confid_thresh = gt_confid_thresh
        max_area_thresh = max_area_thresh_gt

    else:
        confid_thresh = mpeg_confid_thresh
        max_area_thresh = max_area_thresh_mpeg

    result = []
    for b in bboxes:
        b = b.x, b.y, b.w, b.h, b.label, b.conf, b.fid
        (x, y, w, h, label, confid, fid) = b
        if (confid >= confid_thresh and w*h <= max_area_thresh and
                label in relevant_classes):
            result.append(b)
    return result


def iou(b1, b2):
    (x1, y1, w1, h1, label1, confid1, fid) = b1
    (x2, y2, w2, h2, label2, confid2, fid) = b2
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x4 = min(x1+w1, x2+w2)
    y4 = min(y1+h1, y2+h2)
    if x3 > x4 or y3 > y4:
        return 0
    else:
        overlap = (x4-x3)*(y4-y3)
        return overlap/(w1*h1+w2*h2-overlap)


def evaluate(max_fid, map_dd, map_gt, gt_confid_thresh, mpeg_confid_thresh,
             max_area_thresh_gt, max_area_thresh_mpeg, iou_thresh=0.3):
    tp_list = []
    fp_list = []
    fn_list = []
    tp_bb = []
    fp_bb = []
    fn_bb = []
    count_list = []
    for fid in range(max_fid+1):
        bboxes_dd = map_dd[fid]
        bboxes_gt = map_gt[fid]
        bboxes_dd = filter_results(
            bboxes_dd, gt_flag=False, gt_confid_thresh=gt_confid_thresh,
            mpeg_confid_thresh=mpeg_confid_thresh,
            max_area_thresh_gt=max_area_thresh_gt,
            max_area_thresh_mpeg=max_area_thresh_mpeg)
        bboxes_gt = filter_results(
            bboxes_gt, gt_flag=True, gt_confid_thresh=gt_confid_thresh,
            mpeg_confid_thresh=mpeg_confid_thresh,
            max_area_thresh_gt=max_area_thresh_gt,
            max_area_thresh_mpeg=max_area_thresh_mpeg)
        tp = 0
        fp = 0
        fn = 0
        count = 0
        for b_dd in bboxes_dd:
            found = False
            for b_gt in bboxes_gt:
                if iou(b_dd, b_gt) >= iou_thresh:
                    found = True
                    break
            if found:
                tp += 1
                tp_bb.append(b_dd)
            else:
                fp += 1
                fp_bb.append(b_dd)
        for b_gt in bboxes_gt:
            found = False
            for b_dd in bboxes_dd:
                if iou(b_dd, b_gt) >= iou_thresh:
                    found = True
                    break
            if not found:
                fn += 1
                fn_bb.append(b_gt)
            else:
                count += 1
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        count_list.append(count)
    tp = sum(tp_list)
    fp = sum(fp_list)
    fn = sum(fn_list)
    count = sum(count_list)
    return (tp, fp, fn, count,
            round(tp/(tp+fp), 3),
            round(tp/(tp+fn), 3),
            round((2.0*tp/(2.0*tp+fp+fn)), 3),
            tp_bb, fp_bb, fn_bb)


def write_stats_txt(fname, vid_name, config, f1, stats,
                    bw, frames_count, mode, dnn_frames):
    header = ("video-name,low-resolution,high-resolution,low_qp,high_qp,"
              "batch-size,low-threshold,high-threshold,"
              "tracker-length,TP,FP,FN,F1,"
              "low-size,high-size,total-size,frames,dnn_frames,mode")
    stats = (f"{vid_name},{config.low_resolution},{config.high_resolution},"
             f"{config.low_qp},{config.high_qp},{config.batch_size},"
             f"{config.low_threshold},{config.high_threshold},"
             f"{config.tracker_length},{stats[0]},{stats[1]},{stats[2]},"
             f"{f1},{bw[0]},{bw[1]},{bw[0] + bw[1]},"
             f"{frames_count},{str(dnn_frames)},{mode}")

    if not os.path.isfile(fname):
        str_to_write = f"{header}\n{stats}\n"
    else:
        str_to_write = f"{stats}\n"

    with open(fname, "a") as f:
        f.write(str_to_write)


def write_stats_csv(fname, vid_name, config, f1, stats, bw,
                    frames_count, mode, dnn_frames):
    header = ("video-name,low-resolution,high-resolution,low-qp,high-qp,"
              "batch-size,low-threshold,high-threshold,"
              "tracker-length,TP,FP,FN,F1,"
              "low-size,high-size,total-size,frames,dnn_frames,mode").split(",")
    stats = (f"{vid_name},{config.low_resolution},{config.high_resolution},"
             f"{config.low_qp},{config.high_qp},{config.batch_size},"
             f"{config.low_threshold},{config.high_threshold},"
             f"{config.tracker_length},{stats[0]},{stats[1]},{stats[2]},"
             f"{f1},{bw[0]},{bw[1]},{bw[0] + bw[1]},"
             f"{frames_count},{str(dnn_frames)},{mode}").split(",")

    results_files = open(fname, "a")
    csv_writer = csv.writer(results_files)
    if not os.path.isfile(fname):
        # If file does not exist write the header row
        csv_writer.writerow(header)
    csv_writer.writerow(stats)
    results_files.close()


def write_stats(fname, vid_name, config, f1, stats, bw,
                frames_count, mode, dnn_frames):
    if re.match(r"\w+[.]csv\Z", fname):
        write_stats_csv(fname, vid_name, config, f1, stats, bw,
                        frames_count, mode, dnn_frames)
    else:
        write_stats_txt(fname, vid_name, config, f1, stats, bw,
                        frames_count, mode, dnn_frames)


def visualize_stats(tp_bb, fp_bb, fn_bb, images_direc, save_path):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    # process results first into dictionary
    bb_dict = {}
    fids = set()
    for idx, bbs in enumerate([tp_bb, fp_bb, fn_bb]):
        for bb in bbs:
            (x, y, w, h, label, confid, fid) = bb
            if fid not in bb_dict:
                bb_dict[fid] = []
            fids.add(fid)
            bb_dict[fid].append((bb, idx))
    
    for fname in os.listdir(images_direc):
        if "png" not in fname:
            continue
        
        fid = int(fname.split('.')[0])
        if fid not in fids:
            continue

        image_np = cv.imread(
            os.path.join(images_direc, fname))
        width = image_np.shape[1]
        height = image_np.shape[0]

        regions = bb_dict[fid]
        for pair in regions:
            r = pair[0]
            (x, y, w, h, label, confid, bb_fid) = r
            x0 = int(x * width)
            y0 = int(y * height)
            x1 = int(w * width + x0)
            y1 = int(h * height + y0)
            cv.rectangle(image_np, (x0, y0), (x1, y1), colors[pair[1]], 2)
        cv.imwrite(os.path.join(save_path, fname), image_np)


def is_bb_subset(bb, region, context):
    (x, y, w, h, label, confid, bb_fid) = bb
    return (x >= region.x - context and x + w <= region.x + region.w + context and
        y >= region.y - context and y + h <= region.y + region.h + context)


def move_false_negatives(fn_bb, orig_map, bb_dict, context):
    for fn in fn_bb:
        (x, y, w, h, label, confid, bb_fid) = fn
        if bb_fid not in orig_map:
            continue
        orig_to_move = orig_map[bb_fid]
        for check, move_region in orig_to_move.items():
            if check.fid == bb_fid and is_bb_subset(fn, check, context):
                new_x = move_region.x + x - check.x
                new_y = move_region.y + y - check.y
                new_bb = (new_x, new_y, w, h, label, confid, bb_fid)
                if bb_fid not in bb_dict:
                    bb_dict[bb_fid] = {}
                if move_region.fid not in bb_dict[bb_fid]:
                    bb_dict[bb_fid][move_region.fid] = []
                # 2 is color idx of false negative
                bb_dict[bb_fid][move_region.fid].append((new_bb, 2))


def process_tp_or_fp(bbs, color_idx, orig_bb_map, bb_dict):
    for bb in bbs:
        (x, y, w, h, label, confid, bb_fid) = bb
        if bb_fid not in orig_bb_map:
            continue
        orig_bb_to_move = orig_bb_map[bb_fid]
        if bb in orig_bb_to_move:
            move_region = orig_bb_to_move[bb]
            new_bb = (move_region.x, move_region.y, move_region.w, move_region.h, label, confid, bb_fid)
            if bb_fid not in bb_dict:
                bb_dict[bb_fid] = {}
            if move_region.fid not in bb_dict[bb_fid]:
                bb_dict[bb_fid][move_region.fid] = []
            bb_dict[bb_fid][move_region.fid].append((new_bb, color_idx))


def visualize_move_stats(tp_bb, fp_bb, fn_bb, images_direc, orig_bb_map, orig_map, context):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    bb_dict = {}
    move_false_negatives(fn_bb, orig_map, bb_dict, context)
    process_tp_or_fp(tp_bb, 0, orig_bb_map, bb_dict)
    process_tp_or_fp(fp_bb, 1, orig_bb_map, bb_dict)

    for fname in os.listdir(images_direc):
        full_path = os.path.join(images_direc, fname)
        if not os.path.isdir(full_path):
            continue
        
        fname_split = fname.split('-')
        start_idx = int(fname_split[-3])
        end_idx = int(fname_split[-2])

        for im_fname in os.listdir(full_path):
            image_np = cv.imread(
                os.path.join(full_path, im_fname))
            width = image_np.shape[1]
            height = image_np.shape[0]

            move_fid = int(im_fname.split('_')[0])
            
            for orig_fid in range(start_idx, end_idx):
                if orig_fid not in bb_dict:
                    continue
                if move_fid not in bb_dict[orig_fid]:
                    continue

                regions = bb_dict[orig_fid][move_fid]

                for pair in regions:
                    r = pair[0]
                    (x, y, w, h, label, confid, bb_fid) = r
                    x0 = int(x * width)
                    y0 = int(y * height)
                    x1 = int(w * width + x0)
                    y1 = int(h * height + y0)
                    cv.rectangle(image_np, (x0, y0), (x1, y1), colors[pair[1]], 2)
            cv.imwrite(os.path.join(full_path, im_fname), image_np)

def visualize_regions(results, images_direc,
                      low_conf=0.0, high_conf=1.0,
                      label="debugging", save=False,
                      high=False, save_path=None):
    fids = set([r.fid for r in results.regions])
    for fname in os.listdir(images_direc):
        if "png" not in fname:
            continue
        if high:
            fid = int(fname.split('_')[0])
        else:
            fid = int(fname.split('.')[0])
        if fid not in fids:
            continue
        image_np = cv.imread(
            os.path.join(images_direc, fname))
        width = image_np.shape[1]
        height = image_np.shape[0]
        regions = results.regions_dict[fid]
        for r in regions:
            if r.conf < low_conf or r.conf > high_conf:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int(r.w * width + x0)
            y1 = int(r.h * height + y0)
            cv.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
        if save:
            if save_path is None:
                cv.imwrite(
                    os.path.join(images_direc, fname), image_np)
            else:
                cv.imwrite(
                    os.path.join(save_path, fname), image_np)
        else:
            cv.putText(image_np, f"{fid}", (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv.imshow(label, image_np)
            key = cv.waitKey()
            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("k"):
                idx -= 2

    if not save:
        cv.destroyAllWindows()


def visualize_single_regions(region, images_direc, label="debugging"):
    image_path = os.path.join(images_direc, f"{str(region.fid).zfill(10)}.png")
    image_np = cv.imread(image_path)
    width = image_np.shape[1]
    height = image_np.shape[0]

    x0 = int(region.x * width)
    y0 = int(region.y * height)
    x1 = int((region.w * width) + x0)
    y1 = int((region.h * height) + y0)

    cv.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv.putText(image_np, f"{region.fid}, {region.label}, {region.conf:0.2f}, "
               f"{region.w * region.h}",
               (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv.imshow(label, image_np)
    cv.waitKey()
    cv.destroyAllWindows()
