#!/usr/bin/python

import subprocess 
import glob
import json
import os
import numpy as np
import argparse
import cv2
import pdb
import shutil
from detectors import TinyFace


def generate_report(results_dict, report_dir):
    start = '''\\documentclass[a4paper]{article}
\\usepackage{graphicx}
\\usepackage{multirow}
\\usepackage[export]{adjustbox} % for valign
\\usepackage{tikz}
\\begin{document}\n'''

    doc = start
    for key in sorted(results_dict):
        doc += '\section{' + key.replace('_', '\_') + '}\n'
        doc += '\\begin{tabular}{l p{0.5\\textwidth}}\n'
        if os.path.exists(os.path.join(report_dir, key + '.jpg')):
            doc += '\t\\includegraphics[width=0.5\\textwidth, valign=t]{' + key + '.jpg} &\n'
        else:
            doc += '\t\\includegraphics[width=0.5\\textwidth, valign=t]{example-image-a} &\n'
        doc += '\t\tTarget object: ' + results_dict[key]['target_object'] + '\\newline\n'
        doc += '\t\tTarget object present: ' + str(results_dict[key]['is_present']) + '\\newline\n'
        doc += '\t\tIOU: \\textbf{' + '%.4f' % results_dict[key]['iou'] + '}\\newline\n'
        doc += '\t\tResult: \\textbf{' + results_dict[key]['result_type'] + '}\\newline\n'
        doc += '\t\tGT: \\tikz \\fill [green] (0.0,0.0) rectangle (0.3,0.3); Detection: \\tikz \\fill [red] (0.0,0.0) rectangle (0.3,0.3);  \\\\ \n'
        doc += '\\end{tabular}\n\n'

    end = '\\end{document}'
    doc += end
    with open(os.path.join(report_dir, 'root.tex'), 'w') as fp:
        fp.write(doc)

    proc_args = ['make']
    subprocess.run(proc_args, cwd=report_dir)

# Source: http://ronny.rest/tutorials/module/localization_001/iou/
def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


def get_gt_box(gt, width, height):
    gtmin_x = int(round((gt[0] * width) - ((gt[2] * width)/2)))
    gtmax_x = int(round((gt[0] * width) + ((gt[2] * width)/2)))
    gtmin_y = int(round((gt[1] * height) - ((gt[3] * height)/2)))
    gtmax_y = int(round((gt[1] * height) + ((gt[3] * height)/2)))
    gtbox = [gtmin_x, gtmin_y, gtmax_x, gtmax_y]
    return gtbox

def calc_tp_fp(gt, detection, width, height):
    thresholds = np.arange(0.5, 1.0, 0.05)

    gtbox = get_gt_box(gt, width, height)

    dtmin_x = int(detection['min_x'])
    dtmax_x = int(detection['max_x'])
    dtmin_y = int(detection['min_y'])
    dtmax_y = int(detection['max_y'])
    if dtmin_y > dtmax_y:
        dtmin_y, dtmax_y = dtmax_y, dtmin_y
    dtbox = [dtmin_x, dtmin_y, dtmax_x, dtmax_y]

    iou = get_iou(gtbox, dtbox)
    true_pos = iou >= thresholds
    false_pos = iou < thresholds
    return true_pos, false_pos, iou


def is_detection_valid(bbox):
    if len(bbox) != 4:
        return False
    for b in bbox:
        if b < 0:
            return False
        if b > 1:
            return False
    return True

def process_detections(result_files, report_dir, blur_face):
    true_positives = []
    false_positives = []
    false_negatives = 0
    true_negatives = 0

    valid_results = 0
    timeout = 0
    fn_timeout = 0

    results_dict = {}
    if blur_face:
        DET = TinyFace(device='cpu')
    for result_file in result_files:
        folder_path = result_file[:-5]
        if not os.path.exists(folder_path):
            is_object_found = False
            print('No detections for %s' % result_file)
        with open(result_file, 'r') as fp:
            data = json.load(fp)
        if 'object_found' in data['results'].keys():
            is_object_found = data['results']['object_found']
            target_object = data['config']['Target object'][0].lower()
        elif 'person_found' in data['results'].keys():
            is_object_found = data['results']['person_found']
            target_object = 'person'
        else:
            is_object_found = False
            if 'object_detection' in result_file:
                target_object = data['config']['Target object'][0].lower()
            else:
                target_object = 'person'
            print('No detections for %s' % result_file)
        valid_results += 1
        if is_object_found:
            if 'box2d' in data['results'].keys():
                box2d = data['results']['box2d']
            else:
                is_object_found = False # we assume that if no bounding box is given, the object is not detected

        txt_files = sorted(glob.glob(folder_path + '/*.txt'))
        txt_files = [os.path.basename(t) for t in txt_files]
        if 'classes.txt' in txt_files:
            txt_files.remove('classes.txt')
        # this means we have annotations for this image
        if len(txt_files) > 0:
            label_file = txt_files[0] ### TODO: this will not work for Item Delivery
            label = np.loadtxt(os.path.join(folder_path, label_file))
            img_files = sorted(glob.glob(folder_path + '/*.jpg'))
            img_files = [os.path.basename(img) for img in img_files]
            img_file = img_files[0] ### TODO: this will not work for Item Delivery
            img = cv2.imread(os.path.join(folder_path, img_file))
            img_width = img.shape[1]
            img_height = img.shape[0]
        if not os.path.exists(os.path.join(folder_path, 'classes.txt')):
            if 'Target object present' in data['config'].keys():
                if data['config']['Target object present'][0] == "No":
                    found_cls = False
                else:
                    found_cls = True
            else:
                # person detection, there will always be a person?
                found_cls = True
        else:
            with open(os.path.join(folder_path, 'classes.txt')) as fp:
                classes = fp.read().splitlines()
            found_cls = False

            gtboxes = []
            selected_iou = 0.0
            # multiple annotations in this image
            # choose the one with the highest overlap to the detected box
            if label.ndim > 1:
                current_true_pos = None
                current_false_pos = None
                # find the GT object which best matches the detection
                for obj in label:
                    cls_idx = int(obj[0])
                    class_name = classes[cls_idx]
                    if class_name == target_object:
                        found_cls = True
                        gt = obj[1:]
                        gtboxes.append(get_gt_box(gt, img_width, img_height))
                        if is_object_found:
                            true_pos, false_pos, iou = calc_tp_fp(gt, box2d, img_width, img_height)
                            if current_true_pos is None:
                                current_true_pos = true_pos
                                current_false_pos = false_pos
                                selected_iou = iou
                            elif np.sum(true_pos) > np.sum(current_true_pos):
                                current_true_pos = true_pos
                                current_false_pos = false_pos
                                selected_iou = iou
                true_pos = current_true_pos
                false_pos = current_false_pos
            else:
                obj = label
                cls_idx = int(obj[0])
                class_name = classes[cls_idx]
                if class_name == target_object:
                    found_cls = True
                    gt = obj[1:]
                    gtboxes.append(get_gt_box(gt, img_width, img_height))
                    if is_object_found:
                        true_pos, false_pos, iou = calc_tp_fp(gt, box2d, img_width, img_height)
                        selected_iou = iou

        if data['timeout'] is True:
            if found_cls:
                fn_timeout += 1
                false_negatives += 1
                print('%s: FN (timeout)' % os.path.basename(result_file))
                result_type = 'False Negative (timeout)'
            else:
                timeout += 1
                print('%s: timeout' % os.path.basename(result_file))
                result_type = 'Timeout'
        elif not found_cls and is_object_found:
            true_positives.append([False] * 10)
            false_positives.append([True] * 10)
            result_type = 'False Positive'
            print('%s: FP' % os.path.basename(result_file))
        elif found_cls and is_object_found:
            true_positives.append(true_pos)
            false_positives.append(false_pos)
            if selected_iou >= 0.5:
                print('%s: TP' % os.path.basename(result_file))
                result_type = 'True Positive'
            else:
                print('%s: FP' % os.path.basename(result_file))
                result_type = 'False Positive'
        elif found_cls and not is_object_found:
            false_negatives += 1
            print('%s: FN' % os.path.basename(result_file))
            result_type = 'False Negative'
        elif not found_cls and not is_object_found:
            true_negatives += 1
            print('%s: TN' % os.path.basename(result_file))
            result_type = 'True Negative'

        if os.path.exists(folder_path):
            img = cv2.imread(sorted(glob.glob(folder_path + '/*.jpg'))[0], cv2.IMREAD_COLOR)
            if target_object == 'person' and blur_face:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bboxes = DET.detect_faces(gray, conf_th=0.9, scales=[0.5, 1])
                for box in bboxes:
                    fx = max(0, int(box[0] - 10))
                    fy = max(0, int(box[1] - 10))
                    fxx = min(img.shape[1], int(box[2] + 10))
                    fyy = min(img.shape[0], int(box[3] + 10))
                    ff = img[fy:fyy, fx:fxx]
                    ksize = 51 # 21
                    ff = cv2.GaussianBlur(ff, (ksize,ksize), 0)
                    img[fy:fyy, fx:fxx] = ff

            orig = img.copy()
            if is_object_found:
                overlay = orig.copy()
                cv2.rectangle(overlay, (box2d['min_x'], box2d['min_y']), (box2d['max_x'], box2d['max_y']), (0, 0, 255), cv2.FILLED)
                alpha = 0.5
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, gamma=0)
            if gtboxes:
                overlay = orig.copy()
                for box in gtboxes:
                    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), cv2.FILLED)
                alpha = 0.5
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, gamma=0)
            gtboxes = []
            res = os.path.basename(result_file)[:-5]
            img_path = os.path.join(report_dir, res + '.jpg')
            cv2.imwrite(img_path, img)
            results_dict[res] = {'iou': selected_iou, 'result_type': result_type, 'target_object': target_object, 'is_present': found_cls}
            selected_iou = 0.0
        else:
            res = os.path.basename(result_file)[:-5]
            results_dict[res] = {'iou': 0.0, 'result_type': result_type, 'target_object': target_object, 'is_present': found_cls}
    with open(os.path.join(report_dir, 'result.json'), 'w') as fp:
        json.dump(results_dict, fp)

    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)
    avg_true_positives = np.sum(true_positives, axis=1) / float(true_positives.shape[1])
    avg_false_positives = np.sum(false_positives, axis=1) / float(false_positives.shape[1])
    total_tp = np.sum(avg_true_positives)
    total_fp = np.sum(avg_false_positives)
    tptn = total_tp + true_negatives

    print('\n\n')
    print('Total runs: %d' % valid_results)
    print('Total timeouts ', timeout + fn_timeout)

    print('True positives: %.5f' % total_tp)
    print('True negatives: %d' % true_negatives)
    print('False positives: %.5f' % total_fp)
    print('False negatives: %d of which %d were timeouts' % (false_negatives, fn_timeout))
    print('\n\n')

    return results_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str)
    parser.add_argument('--labelImg_path', type=str, default='~/labelImg/labelImg.py')
    parser.add_argument('--report_name', type=str, default='root')
    parser.add_argument('--no-report', dest='generate_report', action='store_false')
    parser.add_argument('--no-face-blur', dest='blur_face', action='store_false')
    parser.set_defaults(generate_report=True)
    parser.set_defaults(blur_face=True)

    args = parser.parse_args()

    results_dir = args.results_dir
    labelImg_exec = os.path.expanduser(args.labelImg_path)
    if not os.path.exists(results_dir):
        print('%s does not exist' % results_dir)
        exit(0)
    print('Annotating and analyzing results in %s' % results_dir)

    classes_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'predefined_classes.txt')

    ### Annotate
    proc_args = [labelImg_exec, results_dir, classes_path]
    subprocess.run(proc_args)
    print('\n\n')

    result_files = sorted(glob.glob(results_dir + '/*.json'))

    report_dir = os.path.join(results_dir, 'report')
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.mkdir(report_dir)

    ### Process Results / Compute Metrics
    results_dict = process_detections(result_files, report_dir, args.blur_face)

    ### Generate Report
    if args.generate_report:
        shutil.copyfile('Makefile', os.path.join(report_dir, 'Makefile'))
        generate_report(results_dict, report_dir)
        shutil.copyfile(os.path.join(report_dir, 'root.pdf'), os.path.join(results_dir, args.report_name + '.pdf'))
        open_pdf = ['evince', '%s.pdf' % args.report_name]
        subprocess.Popen(open_pdf, start_new_session=True, cwd=results_dir)



if __name__ == "__main__":
    main()
